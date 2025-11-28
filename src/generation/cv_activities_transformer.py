"""
CV Activities Transformer.

This module transforms occupation activities from CV_DATA into
achievement-focused CV responsibility bullets.

Run: Used by persona generation pipeline
"""
import sys
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.queries import get_activities_by_occupation, get_occupation_by_id
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


def filter_activities_by_career_level(
    activities: List[str],
    career_level: str
) -> List[str]:
    """
    Filter activities based on career level focus.
    
    Args:
        activities: List of activity strings.
        career_level: Career level (junior, mid, senior, lead).
    
    Returns:
        Filtered list of activities matching career level.
    """
    if not activities:
        return []
    
    # Keywords for different career levels
    level_keywords = {
        "junior": [
            "durchführen", "unterstützen", "erstellen", "bearbeiten",
            "ausführen", "mitarbeiten", "helfen", "assistieren"
        ],
        "mid": [
            "planen", "organisieren", "koordinieren", "durchführen",
            "entwickeln", "umsetzen", "verantworten"
        ],
        "senior": [
            "leiten", "entwickeln", "planen", "koordinieren",
            "verantworten", "optimieren", "strategisch"
        ],
        "lead": [
            "leiten", "führen", "strategisch", "entwickeln",
            "verantworten", "management", "team"
        ]
    }
    
    keywords = level_keywords.get(career_level, level_keywords["mid"])
    
    # Score activities based on keyword matches
    scored_activities = []
    for activity in activities:
        activity_lower = activity.lower()
        score = sum(1 for kw in keywords if kw in activity_lower)
        if score > 0:
            scored_activities.append((activity, score))
    
    # Sort by score (highest first)
    scored_activities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top activities
    filtered = [act for act, score in scored_activities]
    
    # If no matches, return all activities
    if not filtered:
        return activities
    
    return filtered


def transform_activity_to_bullet(
    activity_text: str,
    career_level: str,
    company: str,
    language: str = "de",
    use_ai: bool = True
) -> str:
    """
    Transform activity text to achievement-focused CV bullet.
    
    Args:
        activity_text: Original activity text.
        career_level: Career level for context.
        company: Company name for context.
        language: Language (de, fr, it).
        use_ai: Whether to use AI transformation.
    
    Returns:
        Polished bullet point.
    """
    if not activity_text:
        return ""
    
    # If AI not available or disabled, use simple transformation
    if not use_ai or not OPENAI_AVAILABLE:
        return simple_transform_activity(activity_text, career_level)
    
    try:
        # Create prompt for AI transformation
        prompt = f"""Transform this Swiss occupation activity into a professional CV achievement bullet point.

Activity: {activity_text}
Career Level: {career_level}
Company Context: {company}
Language: {language}

Requirements:
- Achievement-focused (not just task description)
- Include metrics/numbers if possible (e.g., "über 50 Projekte", "3 Jahre", "20% Steigerung")
- Professional tone
- 1 sentence, max 120 characters
- Language: {language}

Return only the bullet point text, no markdown, no explanation."""

        messages = [
            {
                "role": "system",
                "content": "You are a professional CV writer. Transform occupation activities into achievement-focused bullet points with metrics."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Try modern OpenAI client
        if hasattr(_openai_client, 'chat') and callable(getattr(_openai_client, 'chat', None)):
            response = _openai_client.chat.completions.create(
                model=settings.openai_model_mini,
                messages=messages,
                temperature=settings.ai_temperature_creative,
                max_tokens=150
            )
            bullet = response.choices[0].message.content.strip()
        else:
            # Fallback to legacy client
            import openai
            response = openai.ChatCompletion.create(
                model=settings.openai_model_mini,
                messages=messages,
                temperature=settings.ai_temperature_creative,
                max_tokens=150
            )
            bullet = response.choices[0].message.content.strip()
        
        # Clean up bullet (remove markdown, ensure proper format)
        bullet = bullet.replace("*", "").replace("-", "").strip()
        if bullet.startswith("•"):
            bullet = bullet[1:].strip()
        
        return bullet
        
    except Exception as e:
        # Fallback to simple transformation
        return simple_transform_activity(activity_text, career_level)


def simple_transform_activity(activity_text: str, career_level: str) -> str:
    """
    Simple transformation without AI.
    
    Args:
        activity_text: Original activity text.
        career_level: Career level.
    
    Returns:
        Transformed bullet point.
    """
    # Basic transformations
    bullet = activity_text.strip()
    bullet_lower = bullet.lower()
    
    # Remove existing prefixes to avoid duplication
    prefixes_to_remove = ["verantwortung für", "erfolgreich", "verantwortlich für"]
    for prefix in prefixes_to_remove:
        if bullet_lower.startswith(prefix):
            bullet = bullet[len(prefix):].strip()
            bullet_lower = bullet.lower()
    
    # Add achievement language based on career level
    if career_level in ["senior", "lead"]:
        # Add leadership/impact language if not already present
        if "leiten" not in bullet_lower and "führen" not in bullet_lower and "verantwortung" not in bullet_lower:
            bullet = f"Verantwortung für {bullet.lower()}"
        elif "erfolgreich" not in bullet_lower:
            bullet = f"Erfolgreich {bullet.lower()}"
    elif career_level == "mid":
        # Mid level: can have some responsibility language
        if "erfolgreich" not in bullet_lower and random.random() < 0.3:
            bullet = f"Erfolgreich {bullet.lower()}"
    
    # Ensure it starts with capital letter
    if bullet:
        bullet = bullet[0].upper() + bullet[1:] if len(bullet) > 1 else bullet.upper()
    
    return bullet


def extract_activities_from_occupation(job_id: Optional[str]) -> List[str]:
    """
    Extract activities from occupation document.
    
    Args:
        job_id: Occupation job_id.
    
    Returns:
        List of activity strings.
    """
    if not job_id:
        return []
    
    occupation_doc = get_occupation_by_id(job_id)
    if not occupation_doc:
        return []
    
    activities = []
    taetigkeiten = occupation_doc.get("taetigkeiten", {})
    kategorien = taetigkeiten.get("kategorien", {})
    
    if isinstance(kategorien, dict):
        # kategorien is a dict with category names as keys
        for category_name, activity_list in kategorien.items():
            if isinstance(activity_list, list):
                activities.extend(activity_list)
    elif isinstance(kategorien, list):
        # kategorien is a list
        activities = kategorien
    
    # Also try get_activities_by_occupation as fallback
    if not activities:
        activities = get_activities_by_occupation(job_id) or []
    
    return activities


def generate_responsibilities_from_activities(
    job_id: Optional[str],
    career_level: str,
    company: str,
    language: str = "de",
    num_bullets: int = 4,
    is_current_job: bool = True
) -> List[str]:
    """
    Generate responsibility bullets from CV_DATA activities.
    
    Args:
        job_id: Occupation job_id.
        career_level: Career level (junior, mid, senior, lead).
        company: Company name.
        language: Language (de, fr, it).
        num_bullets: Number of bullets to generate.
        is_current_job: Whether this is the current job.
    
    Returns:
        List of responsibility bullet points.
    """
    responsibilities = []
    
    # Extract activities from CV_DATA
    activities = extract_activities_from_occupation(job_id)
    
    if not activities:
        # Fallback: generate generic responsibilities
        return generate_generic_responsibilities(career_level, num_bullets, language)
    
    # Filter activities by career level
    filtered_activities = filter_activities_by_career_level(activities, career_level)
    
    # If not enough filtered, use all activities
    if len(filtered_activities) < num_bullets:
        filtered_activities = activities
    
    # Select activities (fewer for older positions)
    if not is_current_job:
        num_bullets = max(2, num_bullets - 1)  # Fewer bullets for previous jobs
    
    # Select 1-2 activities per category if possible
    selected_activities = []
    if len(filtered_activities) >= num_bullets:
        selected_activities = random.sample(
            filtered_activities,
            min(num_bullets, len(filtered_activities))
        )
    else:
        selected_activities = filtered_activities
    
    # Transform each activity to bullet
    for activity in selected_activities:
        bullet = transform_activity_to_bullet(
            activity,
            career_level,
            company,
            language,
            use_ai=True
        )
        
        if bullet:
            responsibilities.append(bullet)
    
    # Ensure progression
    responsibilities = ensure_progression_in_bullets(
        responsibilities,
        career_level,
        is_older_job=not is_current_job
    )
    
    # If we don't have enough bullets, add generic ones
    while len(responsibilities) < num_bullets:
        generic = generate_generic_responsibility(career_level, language)
        if generic not in responsibilities:
            responsibilities.append(generic)
    
    return responsibilities[:num_bullets]


def generate_generic_responsibilities(
    career_level: str,
    num_bullets: int,
    language: str = "de"
) -> List[str]:
    """
    Generate generic responsibilities when no activities available.
    
    Args:
        career_level: Career level.
        num_bullets: Number of bullets.
        language: Language.
    
    Returns:
        List of generic responsibility bullets.
    """
    generic_templates = {
        "junior": [
            "Durchführung von operativen Aufgaben im Bereich",
            "Unterstützung bei Projekten und täglichen Arbeiten",
            "Mitarbeit in interdisziplinären Teams",
            "Erstellung von Dokumentationen und Berichten"
        ],
        "mid": [
            "Planung und Durchführung von Projekten",
            "Koordination von Arbeitsabläufen",
            "Entwicklung und Umsetzung von Lösungen",
            "Zusammenarbeit mit internen und externen Partnern"
        ],
        "senior": [
            "Leitung von komplexen Projekten",
            "Strategische Planung und Entwicklung",
            "Mentoring von Team-Mitgliedern",
            "Optimierung von Prozessen und Abläufen"
        ],
        "lead": [
            "Strategische Führung und Entwicklung",
            "Leitung von Teams und Projekten",
            "Verantwortung für Budget und Ressourcen",
            "Entwicklung von langfristigen Strategien"
        ]
    }
    
    templates = generic_templates.get(career_level, generic_templates["mid"])
    return random.sample(templates, min(num_bullets, len(templates)))


def generate_generic_responsibility(
    career_level: str,
    language: str = "de"
) -> str:
    """
    Generate a single generic responsibility.
    
    Args:
        career_level: Career level.
        language: Language.
    
    Returns:
        Generic responsibility bullet.
    """
    generic = {
        "junior": "Durchführung von operativen Aufgaben",
        "mid": "Planung und Durchführung von Projekten",
        "senior": "Leitung von komplexen Projekten",
        "lead": "Strategische Führung und Entwicklung"
    }
    
    return generic.get(career_level, generic["mid"])


def ensure_progression_in_bullets(
    bullets: List[str],
    career_level: str,
    is_older_job: bool = False
) -> List[str]:
    """
    Ensure bullets show appropriate progression.
    
    Args:
        bullets: List of bullet points.
        career_level: Career level.
        is_older_job: Whether this is an older position.
    
    Returns:
        Adjusted bullets showing progression.
    """
    if not bullets:
        return bullets
    
    adjusted_bullets = []
    
    for bullet in bullets:
        bullet_lower = bullet.lower()
        
        # For older jobs, simplify language
        if is_older_job:
            # Remove complex/leadership terms if career level was lower
            if career_level in ["junior", "mid"]:
                # Simplify to basic execution language
                if "strategisch" in bullet_lower:
                    bullet = bullet.replace("strategisch", "").replace("Strategisch", "").strip()
                if "leitung" in bullet_lower and career_level == "junior":
                    bullet = bullet.replace("Leitung", "Unterstützung").replace("leitung", "unterstützung")
        
        # For recent jobs, ensure complexity matches career level
        else:
            if career_level in ["senior", "lead"]:
                # Ensure leadership/strategic language
                if "leiten" not in bullet_lower and "führen" not in bullet_lower:
                    if "planen" in bullet_lower or "entwickeln" in bullet_lower:
                        # Already has some complexity
                        pass
                    else:
                        # Add leadership context
                        bullet = f"Verantwortung für {bullet.lower()}"
        
        adjusted_bullets.append(bullet)
    
    return adjusted_bullets

