"""
CV Job History Generator.

This module generates realistic job histories for personas based on:
- Years of experience
- Career progression from CV_DATA
- Skills and technologies
- Company sampling

Run: Used by persona generation pipeline
"""
import sys
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.queries import (
    get_occupation_by_id,
    get_skills_by_occupation,
    get_activities_by_occupation,
    sample_company_by_canton_and_industry
)
from src.database.mongodb_manager import get_db_manager
from src.config import get_settings

settings = get_settings()


def calculate_number_of_previous_jobs(years_experience: int) -> int:
    """
    Calculate number of previous jobs based on years of experience.
    
    Args:
        years_experience: Years of work experience.
    
    Returns:
        Number of previous jobs (excluding current).
    """
    if years_experience <= 2:
        return random.randint(0, 1)
    elif years_experience <= 6:
        return random.randint(1, 2)
    elif years_experience <= 11:
        return random.randint(2, 3)
    else:
        return random.randint(3, 4)


def get_career_level_title(base_title: str, career_level: str) -> str:
    """
    Add career level prefix to occupation title.
    
    Args:
        base_title: Base occupation title.
        career_level: Career level (junior, mid, senior, lead).
    
    Returns:
        Title with career level prefix.
    """
    if career_level == "junior":
        return base_title
    elif career_level == "mid":
        return base_title
    elif career_level == "senior":
        return f"Senior {base_title}"
    elif career_level == "lead":
        # Use "Leiter" for German, "Lead" for English
        if "Leiter" in base_title or "Manager" in base_title:
            return base_title
        else:
            return f"Leiter {base_title}" if random.random() < 0.5 else f"Lead {base_title}"
    else:
        return base_title


def calculate_job_duration(years_experience: int, is_current: bool = False) -> Tuple[int, int]:
    """
    Calculate realistic job duration.
    
    Args:
        years_experience: Total years of experience.
        is_current: Whether this is the current job.
    
    Returns:
        Tuple of (start_year, end_year) or (start_year, None) for current.
    """
    current_year = datetime.now().year
    
    if is_current:
        # Current job: 2-5 years typical
        duration = random.randint(2, 5)
        start_year = current_year - duration
        return start_year, None
    else:
        # Previous jobs: 1.5-3 years typical
        duration = random.randint(1, 3)
        # For previous jobs, we'll calculate backwards
        return duration, None  # Will be calculated in context


def select_activities_for_career_level(
    activities: List[str],
    career_level: str,
    count: int = 4
) -> List[str]:
    """
    Select activities matching career level.
    
    Args:
        activities: List of all activities.
        career_level: Career level.
        count: Number of activities to select.
    
    Returns:
        Selected activities.
    """
    if not activities:
        return []
    
    # Filter activities based on career level
    # Junior/Mid: More operational tasks
    # Senior/Lead: More strategic/leadership tasks
    
    if career_level in ["junior", "mid"]:
        # Prefer operational activities
        operational_keywords = ["durchführen", "erstellen", "unterstützen", "bearbeiten", "umsetzen"]
        filtered = [
            a for a in activities
            if any(kw in a.lower() for kw in operational_keywords)
        ]
        if not filtered:
            filtered = activities
    else:
        # Senior/Lead: Prefer strategic activities
        strategic_keywords = ["leiten", "planen", "entwickeln", "koordinieren", "verantworten"]
        filtered = [
            a for a in activities
            if any(kw in a.lower() for kw in strategic_keywords)
        ]
        if not filtered:
            filtered = activities
    
    # Select random activities
    selected = random.sample(filtered, min(count, len(filtered)))
    
    # If we need more, fill with remaining activities
    if len(selected) < count:
        remaining = [a for a in activities if a not in selected]
        selected.extend(random.sample(remaining, min(count - len(selected), len(remaining))))
    
    return selected[:count]


def get_technologies_from_skills(job_id: Optional[str], limit: int = 8) -> List[str]:
    """
    Get top technologies from occupation skills.
    
    Args:
        job_id: Occupation job_id.
        limit: Maximum number of technologies.
    
    Returns:
        List of technology names.
    """
    if not job_id:
        return []
    
    skills = get_skills_by_occupation(job_id)
    
    # Filter for technical skills
    technical_skills = [
        s.get("skill_name_de", "")
        for s in skills
        if s.get("skill_category") == "technical"
    ]
    
    # Sort by importance (if available)
    technical_skills.sort(
        key=lambda x: next(
            (s.get("importance", 0) for s in skills if s.get("skill_name_de") == x),
            0
        ),
        reverse=True
    )
    
    return technical_skills[:limit]


def get_older_technologies(technologies: List[str], years_ago: int) -> List[str]:
    """
    Get older versions of technologies for historical positions.
    
    Args:
        technologies: Current technologies.
        years_ago: How many years ago.
    
    Returns:
        List of older technology names.
    """
    # Simple mapping for common technologies
    technology_evolution = {
        "Python": ["Python 2.7", "Python 2"],
        "JavaScript": ["ES5", "jQuery", "JavaScript"],
        "React": ["jQuery", "Backbone.js", "AngularJS"],
        "Vue.js": ["jQuery", "Backbone.js"],
        "TypeScript": ["JavaScript", "ES5"],
        "Docker": ["VirtualBox", "VMware"],
        "Kubernetes": ["Docker", "Docker Compose"],
        "AWS": ["On-premise", "Private Cloud"],
        "Git": ["SVN", "CVS"],
        "PostgreSQL": ["MySQL", "PostgreSQL 9"],
        "MongoDB": ["MySQL", "PostgreSQL"],
    }
    
    older_techs = []
    for tech in technologies[:5]:  # Limit to 5
        for current, older_list in technology_evolution.items():
            if current.lower() in tech.lower():
                # Select appropriate older version
                if years_ago > 5:
                    older_techs.append(older_list[-1] if older_list else tech)
                else:
                    older_techs.append(older_list[0] if older_list else tech)
                break
        else:
            # No evolution found, use as-is
            older_techs.append(tech)
    
    return older_techs[:5]


def map_career_level_to_qualification(career_level: str, occupation_doc: Optional[Dict[str, Any]] = None) -> str:
    """
    Map career level to appropriate qualification from career_progression.
    
    Args:
        career_level: Career level (junior, mid, senior, lead).
        occupation_doc: Occupation document from CV_DATA.
    
    Returns:
        Qualification description.
    """
    if not occupation_doc:
        return ""
    
    weiterbildung = occupation_doc.get("weiterbildung", {})
    career_progression = weiterbildung.get("career_progression", [])
    
    if not career_progression or not isinstance(career_progression, list):
        return ""
    
    # Map career level to progression level
    level_mapping = {
        "junior": 0,  # Base occupation
        "mid": 1,     # After BP or experience
        "senior": 2,  # After HFP or experience
        "lead": 3     # After HF/FH or extensive experience
    }
    
    target_level = level_mapping.get(career_level, 0)
    
    if target_level < len(career_progression):
        progression_item = career_progression[target_level]
        if isinstance(progression_item, dict):
            return progression_item.get("type", "") or progression_item.get("title", "")
    
    return ""


def generate_job_history(
    persona: Dict[str, Any],
    occupation_doc: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Generate job history for a persona.
    
    Args:
        persona: Persona dictionary with age, years_experience, job_id, company, etc.
        occupation_doc: Optional occupation document from CV_DATA.
    
    Returns:
        List of job entries with structure:
        {
            "company": str,
            "position": str,
            "location": str,
            "start_date": str (YYYY-MM),
            "end_date": Optional[str] (YYYY-MM or None for current),
            "is_current": bool,
            "responsibilities": List[str],
            "technologies": List[str],
            "category": str
        }
    """
    job_history = []
    
    years_experience = persona.get("years_experience", 0)
    career_level = persona.get("career_level", "mid")
    industry = persona.get("industry", "other")
    canton = persona.get("canton", "ZH")
    job_id = persona.get("job_id")
    current_company = persona.get("company", "Acme AG")
    occupation_title = persona.get("occupation", persona.get("current_title", "Engineer"))
    
    # Get occupation document if not provided
    if not occupation_doc and job_id:
        occupation_doc = get_occupation_by_id(job_id)
    
    # Get activities and skills
    activities = get_activities_by_occupation(job_id) if job_id else []
    current_technologies = get_technologies_from_skills(job_id, limit=8)
    
    # Calculate number of previous jobs
    num_previous_jobs = calculate_number_of_previous_jobs(years_experience)
    
    current_year = datetime.now().year
    
    # 1. CURRENT JOB (most recent)
    current_job_duration = random.randint(2, 5)
    current_start_year = current_year - current_job_duration
    
    # Get current job position with career level
    current_position = get_career_level_title(occupation_title, career_level)
    
    # Select responsibilities for current job
    current_responsibilities = select_activities_for_career_level(activities, career_level, count=4)
    
    # If no activities, create generic ones
    if not current_responsibilities:
        if career_level in ["junior", "mid"]:
            current_responsibilities = [
                f"Durchführung von {occupation_title.lower()} Aufgaben",
                f"Unterstützung bei Projekten",
                f"Zusammenarbeit im Team"
            ]
        else:
            current_responsibilities = [
                f"Leitung von {occupation_title.lower()} Projekten",
                f"Entwicklung von Strategien",
                f"Koordination von Teams"
            ]
    
    current_job = {
        "company": current_company,
        "position": current_position,
        "location": canton,
        "start_date": f"{current_start_year}-01",
        "end_date": None,
        "is_current": True,
        "responsibilities": current_responsibilities[:4],
        "technologies": current_technologies[:8],
        "category": industry
    }
    
    job_history.append(current_job)
    
    # 2. PREVIOUS JOBS (if any)
    remaining_years = years_experience - current_job_duration
    previous_career_levels = []
    
    # Determine career progression for previous jobs
    if career_level == "lead":
        previous_career_levels = ["senior", "mid", "junior"]
    elif career_level == "senior":
        previous_career_levels = ["mid", "junior"]
    elif career_level == "mid":
        previous_career_levels = ["junior"]
    else:
        previous_career_levels = []
    
    # Generate previous jobs
    for i in range(num_previous_jobs):
        if remaining_years <= 0:
            break
        
        # Determine career level for this job
        if i < len(previous_career_levels):
            job_career_level = previous_career_levels[i]
        else:
            job_career_level = "junior"
        
        # Calculate job duration (1.5-3 years)
        job_duration = random.randint(1, 3)
        if job_duration > remaining_years:
            job_duration = remaining_years
        
        if job_duration <= 0:
            break
        
        # Calculate dates
        job_end_year = current_start_year - 1
        job_start_year = job_end_year - job_duration + 1
        
        # Sample company (same or related industry)
        previous_company_doc = sample_company_by_canton_and_industry(canton, industry)
        previous_company = previous_company_doc.get("name") if previous_company_doc else f"Company {i+1} AG"
        
        # Get position for this career level
        previous_position = get_career_level_title(occupation_title, job_career_level)
        
        # Select responsibilities (fewer than current)
        num_responsibilities = random.randint(2, 3)
        previous_responsibilities = select_activities_for_career_level(
            activities, job_career_level, count=num_responsibilities
        )
        
        if not previous_responsibilities:
            previous_responsibilities = [
                f"Durchführung von {occupation_title.lower()} Aufgaben"
            ]
        
        # Get older technologies
        years_ago = current_year - job_end_year
        previous_technologies = get_older_technologies(current_technologies, years_ago)
        
        previous_job = {
            "company": previous_company,
            "position": previous_position,
            "location": canton,
            "start_date": f"{job_start_year}-01",
            "end_date": f"{job_end_year}-12",
            "is_current": False,
            "responsibilities": previous_responsibilities,
            "technologies": previous_technologies[:5],
            "category": industry
        }
        
        job_history.append(previous_job)
        
        # Update for next iteration
        current_start_year = job_start_year - 1
        remaining_years -= job_duration
    
    # Sort by start_date (most recent first)
    job_history.sort(key=lambda x: x.get("start_date", ""), reverse=True)
    
    return job_history


def validate_job_history(
    job_history: List[Dict[str, Any]],
    persona_age: int,
    years_experience: int
) -> List[Dict[str, Any]]:
    """
    Validate job history timeline consistency.
    
    Args:
        job_history: List of job entries.
        persona_age: Persona's current age.
        years_experience: Years of work experience.
    
    Returns:
        Validated job history.
    """
    if not job_history:
        return job_history
    
    current_year = datetime.now().year
    birth_year = current_year - persona_age
    
    # Work typically starts around age 18-22
    min_work_start_age = 18
    max_work_start_age = 22
    
    validated = []
    
    for job in job_history:
        start_date = job.get("start_date", "")
        end_date = job.get("end_date")
        is_current = job.get("is_current", False)
        
        if start_date:
            try:
                start_year = int(start_date.split("-")[0])
                
                # Check: Job shouldn't start before minimum work age
                if start_year < birth_year + min_work_start_age:
                    start_year = birth_year + min_work_start_age
                    job["start_date"] = f"{start_year}-01"
                
                # Check: Job shouldn't start after current year
                if start_year > current_year:
                    start_year = current_year - 1
                    job["start_date"] = f"{start_year}-01"
                
                # Validate end_date
                if end_date:
                    end_year = int(end_date.split("-")[0])
                    if end_year < start_year:
                        end_year = start_year + 1
                    if end_year > current_year:
                        end_year = current_year
                    job["end_date"] = f"{end_year}-12"
                elif not is_current:
                    # Previous job should have end_date
                    start_year = int(job["start_date"].split("-")[0])
                    end_year = min(start_year + 2, current_year - 1)
                    job["end_date"] = f"{end_year}-12"
                
            except (ValueError, IndexError):
                # Invalid date format, skip validation
                pass
        
        validated.append(job)
    
    return validated


def get_job_history_summary(job_history: List[Dict[str, Any]]) -> str:
    """
    Generate a text summary of job history.
    
    Args:
        job_history: List of job entries.
    
    Returns:
        Formatted job history summary string.
    """
    if not job_history:
        return "Keine Berufserfahrung verfügbar."
    
    summary_parts = []
    
    for job in job_history:
        company = job.get("company", "")
        position = job.get("position", "")
        start_date = job.get("start_date", "")
        end_date = job.get("end_date", "heute")
        
        if company and position:
            start_year = start_date.split("-")[0] if start_date else "?"
            end_year = end_date.split("-")[0] if end_date else "heute"
            summary_parts.append(f"{position} bei {company} ({start_year}-{end_year})")
    
    return " | ".join(summary_parts) if summary_parts else "Keine Berufserfahrung verfügbar."

