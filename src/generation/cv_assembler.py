"""
CV Assembler - Complete CV Document Generator.

This module assembles all CV components into a complete document:
- Personal information
- Portrait
- Summary (AI-generated)
- Education history
- Job history
- Skills
- Additional education
- Languages
- Hobbies

Run: Used by persona generation pipeline
"""
import sys
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from PIL import Image
import io
import base64

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.queries import get_occupation_by_id
from src.generation.cv_education_generator import generate_education_history
from src.generation.cv_job_history_generator import generate_job_history
from src.generation.cv_continuing_education import generate_additional_education
from src.generation.cv_activities_transformer import generate_responsibilities_from_activities
from src.config import get_settings

settings = get_settings()

# OpenAI client setup
OPENAI_AVAILABLE = False
_openai_client = None

try:
    try:
        from openai import OpenAI
        if settings.openai_api_key:
            _openai_client = OpenAI(api_key=settings.openai_api_key)
            OPENAI_AVAILABLE = True
    except ImportError:
        try:
            import openai
            if settings.openai_api_key:
                openai.api_key = settings.openai_api_key
            OPENAI_AVAILABLE = True
        except ImportError:
            pass
except Exception:
    pass


@dataclass
class CVDocument:
    """Complete CV document structure."""
    # Personal
    first_name: str
    last_name: str
    full_name: str
    age: int
    gender: str
    canton: str
    city: Optional[str] = None
    email: str = ""
    phone: str = ""
    address: Optional[str] = None
    portrait_path: Optional[str] = None
    portrait_base64: Optional[str] = None
    
    # Professional
    current_title: str = ""
    industry: str = ""
    career_level: str = ""
    years_experience: int = 0
    
    # Content
    summary: str = ""
    education: List[Dict[str, Any]] = field(default_factory=list)
    jobs: List[Dict[str, Any]] = field(default_factory=list)
    skills: Dict[str, List[str]] = field(default_factory=dict)  # {"technical": [...], "soft": [...], "languages": [...]}
    additional_education: List[Dict[str, Any]] = field(default_factory=list)
    hobbies: List[str] = field(default_factory=list)
    
    # Metadata
    language: str = "de"
    created_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for template rendering."""
        return {
            "personal": {
                "first_name": self.first_name,
                "last_name": self.last_name,
                "full_name": self.full_name,
                "age": self.age,
                "gender": self.gender,
                "canton": self.canton,
                "city": self.city,
                "email": self.email,
                "phone": self.phone,
                "address": self.address,
                "portrait_path": self.portrait_path,
                "portrait_base64": self.portrait_base64
            },
            "professional": {
                "current_title": self.current_title,
                "industry": self.industry,
                "career_level": self.career_level,
                "years_experience": self.years_experience
            },
            "content": {
                "summary": self.summary,
                "education": self.education,
                "jobs": self.jobs,
                "skills": self.skills,
                "additional_education": self.additional_education,
                "hobbies": self.hobbies
            },
            "metadata": {
                "language": self.language,
                "created_at": self.created_at
            }
        }


def load_portrait_image(portrait_path: Optional[str], resize: Tuple[int, int] = (150, 150), circular: bool = False) -> Optional[str]:
    """
    Load and process portrait image.
    
    Args:
        portrait_path: Relative path to portrait image.
        resize: Target size (width, height).
        circular: Whether to apply circular crop.
    
    Returns:
        Base64-encoded image string or None.
    """
    if not portrait_path:
        return None
    
    full_path = project_root / "data" / "portraits" / portrait_path
    
    if not full_path.exists():
        return None
    
    try:
        from PIL import Image
        
        # Load image
        img = Image.open(full_path)
        
        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize
        img = img.resize(resize, Image.Resampling.LANCZOS)
        
        # Circular crop if requested
        if circular:
            # Create circular mask
            mask = Image.new("L", resize, 0)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(mask)
            draw.ellipse([0, 0, resize[0], resize[1]], fill=255)
            
            # Apply mask
            output = Image.new("RGB", resize, (255, 255, 255))
            output.paste(img, (0, 0), mask)
            img = output
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        print(f"Warning: Could not load portrait: {e}")
        return None


def generate_summary(
    persona: Dict[str, Any],
    occupation_doc: Optional[Dict[str, Any]] = None,
    language: str = "de"
) -> str:
    """
    Generate CV summary using AI.
    
    Args:
        persona: Persona dictionary.
        occupation_doc: Occupation document from CV_DATA.
        language: Language (de, fr, it).
    
    Returns:
        Generated summary text (2-3 sentences).
    """
    if not OPENAI_AVAILABLE or not settings.openai_api_key:
        return generate_fallback_summary(persona, language)
    
    # Extract relevant information
    name = f"{persona.get('first_name')} {persona.get('last_name')}"
    age = persona.get("age", 25)
    years_exp = persona.get("years_experience", 0)
    career_level = persona.get("career_level", "mid")
    industry = persona.get("industry", "")
    occupation_title = persona.get("occupation", persona.get("current_title", ""))
    
    # Get description from occupation
    description = ""
    if occupation_doc:
        description = occupation_doc.get("description", "")
        berufsverhaeltnisse = occupation_doc.get("berufsverhaeltnisse", {})
        if isinstance(berufsverhaeltnisse, dict):
            beschreibung = berufsverhaeltnisse.get("beschreibung", "")
            if beschreibung:
                description += " " + beschreibung
    
    # Language-specific prompts
    prompts = {
        "de": f"""Erstelle einen professionellen CV-Zusammenfassungstext (2-3 Sätze) für:

Name: {name}
Alter: {age} Jahre
Berufserfahrung: {years_exp} Jahre
Karrierelevel: {career_level}
Branche: {industry}
Beruf: {occupation_title}

Berufsbeschreibung: {description[:500]}

Anforderungen:
- Professionell und überzeugend
- Zeigt Erfahrung und Kompetenz
- 2-3 Sätze, max 200 Wörter
- Schweizer CV-Stil

Nur den Text zurückgeben, keine Markdown, keine Erklärung.""",
        "fr": f"""Créez un texte de résumé professionnel de CV (2-3 phrases) pour:

Nom: {name}
Âge: {age} ans
Expérience: {years_exp} ans
Niveau de carrière: {career_level}
Secteur: {industry}
Profession: {occupation_title}

Description professionnelle: {description[:500]}

Exigences:
- Professionnel et convaincant
- Montre l'expérience et les compétences
- 2-3 phrases, max 200 mots
- Style CV suisse

Retournez uniquement le texte, pas de markdown, pas d'explication.""",
        "it": f"""Crea un testo di riepilogo professionale del CV (2-3 frasi) per:

Nome: {name}
Età: {age} anni
Esperienza: {years_exp} anni
Livello di carriera: {career_level}
Settore: {industry}
Professione: {occupation_title}

Descrizione professionale: {description[:500]}

Requisiti:
- Professionale e convincente
- Mostra esperienza e competenze
- 2-3 frasi, max 200 parole
- Stile CV svizzero

Restituisci solo il testo, nessun markdown, nessuna spiegazione."""
    }
    
    prompt = prompts.get(language, prompts["de"])
    
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a professional CV writer specializing in Swiss CV formats."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Try modern OpenAI client
        if _openai_client and hasattr(_openai_client, 'chat'):
            response = _openai_client.chat.completions.create(
                model=settings.openai_model_mini,
                messages=messages,
                temperature=settings.ai_temperature_creative,
                max_tokens=200
            )
            summary = response.choices[0].message.content.strip()
        else:
            # Fallback: use simple summary
            summary = generate_fallback_summary(persona, language)
        
        # Clean up
        summary = summary.replace("**", "").replace("*", "").strip()
        
        return summary
        
    except Exception as e:
        print(f"Warning: AI summary generation failed: {e}")
        return generate_fallback_summary(persona, language)


def generate_fallback_summary(persona: Dict[str, Any], language: str = "de") -> str:
    """Generate fallback summary without AI."""
    name = persona.get("first_name", "")
    years_exp = persona.get("years_experience", 0)
    career_level = persona.get("career_level", "mid")
    occupation_title = persona.get("occupation", persona.get("current_title", ""))
    
    summaries = {
        "de": f"{name} ist ein {career_level}-level {occupation_title.lower()} mit {years_exp} Jahren Berufserfahrung. Spezialisiert auf {persona.get('industry', 'verschiedene Bereiche')} mit Fokus auf Qualität und Effizienz.",
        "fr": f"{name} est un {occupation_title.lower()} de niveau {career_level} avec {years_exp} ans d'expérience professionnelle. Spécialisé dans {persona.get('industry', 'divers domaines')} avec un accent sur la qualité et l'efficacité.",
        "it": f"{name} è un {occupation_title.lower()} di livello {career_level} con {years_exp} anni di esperienza professionale. Specializzato in {persona.get('industry', 'vari settori')} con focus su qualità ed efficienza."
    }
    
    return summaries.get(language, summaries["de"])


def generate_hobbies(language: str = "de", use_ai: bool = True) -> List[str]:
    """
    Generate realistic Swiss hobbies.
    
    Args:
        language: Language (de, fr, it).
        use_ai: Whether to use AI generation.
    
    Returns:
        List of hobby strings (3-5 items).
    """
    # Common Swiss hobbies
    swiss_hobbies = {
        "de": [
            "Wandern in den Alpen",
            "Skifahren",
            "Fussball",
            "Velofahren",
            "Lesen",
            "Kochen",
            "Musik",
            "Fotografie",
            "Reisen",
            "Volunteering"
        ],
        "fr": [
            "Randonnée dans les Alpes",
            "Ski",
            "Football",
            "Vélo",
            "Lecture",
            "Cuisine",
            "Musique",
            "Photographie",
            "Voyages",
            "Bénévolat"
        ],
        "it": [
            "Escursioni nelle Alpi",
            "Sci",
            "Calcio",
            "Ciclismo",
            "Lettura",
            "Cucina",
            "Musica",
            "Fotografia",
            "Viaggi",
            "Volontariato"
        ]
    }
    
    if use_ai and OPENAI_AVAILABLE:
        try:
            prompts = {
                "de": "Generiere 4-5 realistische Schweizer Hobbys für einen CV. Rücke nur eine kommagetrennte Liste zurück, keine Erklärung.",
                "fr": "Génère 4-5 loisirs suisses réalistes pour un CV. Retourne uniquement une liste séparée par des virgules, pas d'explication.",
                "it": "Genera 4-5 hobby svizzeri realistici per un CV. Restituisci solo un elenco separato da virgole, nessuna spiegazione."
            }
            
            messages = [
                {"role": "system", "content": "You are a professional CV writer."},
                {"role": "user", "content": prompts.get(language, prompts["de"])}
            ]
            
            if _openai_client and hasattr(_openai_client, 'chat'):
                response = _openai_client.chat.completions.create(
                    model=settings.openai_model_mini,
                    messages=messages,
                    temperature=settings.ai_temperature_creative,
                    max_tokens=100
                )
                hobbies_text = response.choices[0].message.content.strip()
            else:
                # Fallback: use predefined hobbies
                hobbies_text = ""
            
            # Parse hobbies
            hobbies = [h.strip() for h in hobbies_text.split(",") if h.strip()]
            return hobbies[:5] if hobbies else swiss_hobbies.get(language, swiss_hobbies["de"])[:5]
            
        except Exception:
            pass
    
    # Fallback to predefined hobbies
    hobbies_list = swiss_hobbies.get(language, swiss_hobbies["de"])
    return random.sample(hobbies_list, min(5, len(hobbies_list)))


def categorize_skills(skills: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Categorize skills into technical, soft, and languages.
    
    Args:
        skills: List of skill dictionaries from occupation_skills.
    
    Returns:
        Dictionary with categorized skills.
    """
    categorized = {
        "technical": [],
        "soft": [],
        "languages": []
    }
    
    for skill in skills:
        category = skill.get("skill_category", "soft")
        skill_name = skill.get("skill_name_de", "")
        
        if not skill_name:
            continue
        
        if category == "technical":
            categorized["technical"].append(skill_name)
        elif category == "soft":
            categorized["soft"].append(skill_name)
        elif category == "physical":
            # Physical skills can go to technical or soft
            categorized["technical"].append(skill_name)
    
    # Limit to top skills per category
    categorized["technical"] = categorized["technical"][:10]
    categorized["soft"] = categorized["soft"][:8]
    
    return categorized


def get_languages_for_cv(canton: str, primary_language: str) -> List[str]:
    """
    Get languages for CV based on canton and primary language.
    
    Args:
        canton: Canton code.
        primary_language: Primary language (de, fr, it).
    
    Returns:
        List of language strings with proficiency levels.
    """
    languages = []
    
    # Primary language (native or fluent)
    lang_names = {
        "de": "Deutsch",
        "fr": "Französisch",
        "it": "Italienisch",
        "en": "Englisch"
    }
    
    languages.append(f"{lang_names.get(primary_language, primary_language)} (Muttersprache)")
    
    # Add other Swiss languages based on canton
    multilingual_cantons = {
        "BE": ["de", "fr"],
        "VS": ["de", "fr"],
        "FR": ["fr", "de"],
        "GR": ["de", "it", "rm"]
    }
    
    if canton in multilingual_cantons:
        other_langs = multilingual_cantons[canton]
        for lang in other_langs:
            if lang != primary_language:
                proficiency = random.choice(["Fließend", "Gut", "Grundkenntnisse"])
                languages.append(f"{lang_names.get(lang, lang)} ({proficiency})")
    
    # Most Swiss people speak English
    if random.random() < 0.8:  # 80% chance
        english_level = random.choice(["Fließend", "Gut", "Grundkenntnisse"])
        languages.append(f"Englisch ({english_level})")
    
    return languages


def format_date_swiss(date_str: Optional[str], language: str = "de") -> str:
    """
    Format date in Swiss format (DD.MM.YYYY).
    
    Args:
        date_str: Date string (YYYY-MM format).
        language: Language for month names if needed.
    
    Returns:
        Formatted date string.
    """
    if not date_str:
        return ""
    
    try:
        if "-" in date_str:
            parts = date_str.split("-")
            if len(parts) >= 2:
                year = parts[0]
                month = parts[1]
                return f"{month}.{year}"
    except:
        pass
    
    return date_str


def generate_complete_cv(persona: Dict[str, Any]) -> CVDocument:
    """
    Generate complete CV document from persona.
    
    Args:
        persona: Persona dictionary from sampling.
    
    Returns:
        Complete CVDocument object.
    """
    # Load full occupation document
    job_id = persona.get("job_id")
    occupation_doc = get_occupation_by_id(job_id) if job_id else None
    
    # Generate all sections
    language = persona.get("language", "de")
    
    # 1. Personal information
    first_name = persona.get("first_name", "")
    last_name = persona.get("last_name", "")
    full_name = persona.get("full_name", f"{first_name} {last_name}")
    
    # Generate city based on canton
    canton = persona.get("canton", "ZH")
    city = generate_city_for_canton(canton)
    
    # Generate address
    address = f"{city}, {canton}"
    
    # 2. Portrait
    portrait_path = persona.get("portrait_path")
    portrait_base64 = load_portrait_image(portrait_path, resize=(150, 150), circular=True)
    
    # 3. Summary (AI-generated)
    summary = generate_summary(persona, occupation_doc, language)
    
    # 4. Education history
    education_history = generate_education_history(persona, occupation_doc)
    
    # 5. Job history
    job_history = generate_job_history(persona, occupation_doc)
    
    # Add responsibilities to job history
    for job in job_history:
        if job.get("is_current", False):
            # Generate responsibilities for current job
            responsibilities = generate_responsibilities_from_activities(
                job_id,
                persona.get("career_level", "mid"),
                job.get("company", ""),
                language,
                num_bullets=4,
                is_current_job=True
            )
        else:
            # Fewer responsibilities for previous jobs
            previous_level = "mid" if persona.get("career_level") in ["senior", "lead"] else "junior"
            responsibilities = generate_responsibilities_from_activities(
                job_id,
                previous_level,
                job.get("company", ""),
                language,
                num_bullets=2,
                is_current_job=False
            )
        
        job["responsibilities"] = responsibilities
    
    # 6. Skills (categorized)
    skills_list = persona.get("skills", [])
    if isinstance(skills_list, list) and skills_list and isinstance(skills_list[0], str):
        # Skills are already strings
        from src.database.queries import get_skills_by_occupation
        skills_docs = get_skills_by_occupation(job_id) if job_id else []
        categorized_skills = categorize_skills(skills_docs)
    else:
        # Skills are dictionaries
        categorized_skills = categorize_skills(skills_list)
    
    # Add languages to skills
    languages = get_languages_for_cv(canton, language)
    categorized_skills["languages"] = languages
    
    # 7. Additional education
    base_education_end = None
    if education_history:
        base_education_end = education_history[0].get("end_year")
    
    additional_education = generate_additional_education(
        persona,
        occupation_doc,
        base_education_end_year=base_education_end
    )
    
    # 8. Hobbies
    hobbies = generate_hobbies(language, use_ai=True)
    
    # Create CVDocument
    cv_doc = CVDocument(
        first_name=first_name,
        last_name=last_name,
        full_name=full_name,
        age=persona.get("age", 25),
        gender=persona.get("gender", ""),
        canton=canton,
        city=city,
        email=persona.get("email", ""),
        phone=persona.get("phone", ""),
        address=address,
        portrait_path=portrait_path,
        portrait_base64=portrait_base64,
        current_title=persona.get("current_title", persona.get("occupation", "")),
        industry=persona.get("industry", ""),
        career_level=persona.get("career_level", "mid"),
        years_experience=persona.get("years_experience", 0),
        summary=summary,
        education=education_history,
        jobs=job_history,
        skills=categorized_skills,
        additional_education=additional_education,
        hobbies=hobbies,
        language=language,
        created_at=datetime.now().isoformat()
    )
    
    return cv_doc


def generate_city_for_canton(canton: str) -> str:
    """
    Generate realistic city name for canton.
    
    Args:
        canton: Canton code.
    
    Returns:
        City name.
    """
    # Major cities per canton
    canton_cities = {
        "ZH": "Zürich",
        "BE": "Bern",
        "BS": "Basel",
        "GE": "Genève",
        "VD": "Lausanne",
        "AG": "Aarau",
        "SG": "St. Gallen",
        "LU": "Luzern",
        "TI": "Lugano",
        "VS": "Sion",
        "FR": "Fribourg",
        "GR": "Chur",
        "NE": "Neuchâtel",
        "TG": "Frauenfeld",
        "SH": "Schaffhausen",
        "AR": "Herisau",
        "AI": "Appenzell",
        "GL": "Glarus",
        "NW": "Stans",
        "OW": "Sarnen",
        "SZ": "Schwyz",
        "UR": "Altdorf",
        "ZG": "Zug",
        "SO": "Solothurn",
        "BL": "Liestal",
        "JU": "Delémont"
    }
    
    return canton_cities.get(canton, f"City {canton}")


def get_section_headers(language: str) -> Dict[str, str]:
    """
    Get section headers translated for language.
    
    Args:
        language: Language (de, fr, it).
    
    Returns:
        Dictionary with section headers.
    """
    headers = {
        "de": {
            "personal": "Persönliche Angaben",
            "summary": "Zusammenfassung",
            "experience": "Berufserfahrung",
            "education": "Ausbildung",
            "skills": "Kompetenzen",
            "technical_skills": "Technische Kompetenzen",
            "soft_skills": "Persönliche Kompetenzen",
            "languages": "Sprachen",
            "certifications": "Zertifikate & Weiterbildung",
            "hobbies": "Hobbys & Interessen"
        },
        "fr": {
            "personal": "Informations personnelles",
            "summary": "Résumé",
            "experience": "Expérience professionnelle",
            "education": "Formation",
            "skills": "Compétences",
            "technical_skills": "Compétences techniques",
            "soft_skills": "Compétences personnelles",
            "languages": "Langues",
            "certifications": "Certificats & Formation continue",
            "hobbies": "Loisirs & Intérêts"
        },
        "it": {
            "personal": "Informazioni personali",
            "summary": "Riassunto",
            "experience": "Esperienza professionale",
            "education": "Formazione",
            "skills": "Competenze",
            "technical_skills": "Competenze tecniche",
            "soft_skills": "Competenze personali",
            "languages": "Lingue",
            "certifications": "Certificati & Formazione continua",
            "hobbies": "Hobby & Interessi"
        }
    }
    
    return headers.get(language, headers["de"])

