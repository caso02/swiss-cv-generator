"""
CV Timeline Validator.

This module validates and fixes CV timeline consistency:
- Education and job history alignment with persona age
- No overlapping job periods
- No unexplained gaps > 12 months
- Logical career progression
- Realistic age/experience/career_level relationships

Run: Used by CV generation pipeline
"""
import sys
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class TimelineIssue:
    """Represents a timeline validation issue."""
    severity: str  # "error", "warning", "info"
    category: str  # "overlap", "gap", "age", "progression", "duration"
    message: str
    affected_periods: List[Tuple[int, int]] = None  # (start_year, end_year)
    suggested_fix: Optional[str] = None


class ValidationError(Exception):
    """Raised when timeline validation fails with critical errors."""
    def __init__(self, message: str, issues: List[TimelineIssue]):
        self.message = message
        self.issues = issues
        super().__init__(self.message)


def parse_date_to_year(date_str: Optional[str]) -> Optional[int]:
    """
    Parse date string (YYYY-MM) to year.
    
    Args:
        date_str: Date string in format "YYYY-MM" or None.
    
    Returns:
        Year as integer or None.
    """
    if not date_str:
        return None
    
    try:
        if "-" in date_str:
            return int(date_str.split("-")[0])
        else:
            return int(date_str)
    except (ValueError, AttributeError):
        return None


def year_to_date_string(year: int, month: int = 1) -> str:
    """
    Convert year to date string format.
    
    Args:
        year: Year as integer.
        month: Month (1-12), default 1.
    
    Returns:
        Date string in format "YYYY-MM".
    """
    return f"{year}-{month:02d}"


def calculate_total_job_years(job_history: List[Dict[str, Any]]) -> float:
    """
    Calculate total years of work experience from job history.
    
    Args:
        job_history: List of job entries.
    
    Returns:
        Total years as float.
    """
    total_years = 0.0
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    for job in job_history:
        start_date = job.get("start_date")
        end_date = job.get("end_date")
        is_current = job.get("is_current", False)
        
        if not start_date:
            continue
        
        start_year = parse_date_to_year(start_date)
        if not start_year:
            continue
        
        if is_current or not end_date:
            # Current job: calculate to present
            end_year = current_year
            end_month = current_month
        else:
            end_year = parse_date_to_year(end_date)
            if not end_year:
                continue
            # Estimate month from date string
            try:
                end_month = int(end_date.split("-")[1]) if "-" in end_date else 6
            except:
                end_month = 6
        
        # Estimate start month
        try:
            start_month = int(start_date.split("-")[1]) if "-" in start_date else 1
        except:
            start_month = 1
        
        # Calculate duration in years
        years = end_year - start_year
        months = end_month - start_month
        total_years += years + (months / 12.0)
    
    return total_years


def check_job_overlaps(job_history: List[Dict[str, Any]]) -> List[TimelineIssue]:
    """
    Check for overlapping job periods.
    
    Args:
        job_history: List of job entries.
    
    Returns:
        List of overlap issues.
    """
    issues = []
    current_year = datetime.now().year
    
    # Sort jobs by start date
    sorted_jobs = sorted(
        job_history,
        key=lambda j: parse_date_to_year(j.get("start_date", "2000-01")) or 2000
    )
    
    for i in range(len(sorted_jobs) - 1):
        job1 = sorted_jobs[i]
        job2 = sorted_jobs[i + 1]
        
        start1 = parse_date_to_year(job1.get("start_date"))
        end1 = parse_date_to_year(job2.get("end_date")) if not job1.get("is_current") else current_year
        
        start2 = parse_date_to_year(job2.get("start_date"))
        end2 = parse_date_to_year(job2.get("end_date")) if not job2.get("is_current") else current_year
        
        if not all([start1, start2]):
            continue
        
        # Check overlap
        if end1 and start2 and end1 > start2:
            overlap_start = start2
            overlap_end = min(end1, end2) if end2 else end1
            
            issues.append(TimelineIssue(
                severity="error",
                category="overlap",
                message=f"Job overlap: {job1.get('company')} ({start1}-{end1}) overlaps with {job2.get('company')} ({start2}-{end2})",
                affected_periods=[(overlap_start, overlap_end)],
                suggested_fix=f"Adjust end date of {job1.get('company')} to {start2 - 1} or start date of {job2.get('company')} to {end1 + 1}"
            ))
    
    return issues


def check_gaps(
    education_history: List[Dict[str, Any]],
    job_history: List[Dict[str, Any]],
    max_gap_months: int = 12
) -> List[TimelineIssue]:
    """
    Check for unexplained gaps > max_gap_months.
    
    Args:
        education_history: List of education entries.
        job_history: List of job entries.
        max_gap_months: Maximum allowed gap in months.
    
    Returns:
        List of gap issues.
    """
    issues = []
    current_year = datetime.now().year
    
    # Get education end year
    education_end = None
    if education_history:
        education_end = max([e.get("end_year", 0) for e in education_history])
    
    # Get first job start
    first_job_start = None
    if job_history:
        sorted_jobs = sorted(
            job_history,
            key=lambda j: parse_date_to_year(j.get("start_date", "2000-01")) or 2000
        )
        first_job_start = parse_date_to_year(sorted_jobs[0].get("start_date"))
    
    # Check gap between education and first job
    if education_end and first_job_start:
        gap_years = first_job_start - education_end
        if gap_years > 0 and gap_years * 12 > max_gap_months:
            issues.append(TimelineIssue(
                severity="warning",
                category="gap",
                message=f"Large gap between education end ({education_end}) and first job ({first_job_start}): {gap_years} years",
                affected_periods=[(education_end, first_job_start)],
                suggested_fix="Insert gap filler: 'Sabbatical / Weiterbildung' or 'Freelance Projekte'"
            ))
    
    # Check gaps between jobs
    sorted_jobs = sorted(
        job_history,
        key=lambda j: parse_date_to_year(j.get("start_date", "2000-01")) or 2000
    )
    
    for i in range(len(sorted_jobs) - 1):
        job1 = sorted_jobs[i]
        job2 = sorted_jobs[i + 1]
        
        end1 = parse_date_to_year(job1.get("end_date")) if not job1.get("is_current") else current_year
        start2 = parse_date_to_year(job2.get("start_date"))
        
        if end1 and start2:
            gap_years = start2 - end1
            if gap_years > 0 and gap_years * 12 > max_gap_months:
                issues.append(TimelineIssue(
                    severity="warning",
                    category="gap",
                    message=f"Gap between {job1.get('company')} ({end1}) and {job2.get('company')} ({start2}): {gap_years} years",
                    affected_periods=[(end1, start2)],
                    suggested_fix="Insert gap filler: 'Elternzeit', 'Sabbatical / Weiterbildung', or 'Freelance Projekte'"
                ))
    
    return issues


def check_career_progression(job_history: List[Dict[str, Any]]) -> List[TimelineIssue]:
    """
    Check for logical career progression (no Senior → Junior regression).
    
    Args:
        job_history: List of job entries.
    
    Returns:
        List of progression issues.
    """
    issues = []
    
    # Career level hierarchy
    level_hierarchy = {
        "junior": 1,
        "mid": 2,
        "senior": 3,
        "lead": 4
    }
    
    # Sort jobs chronologically (oldest first)
    sorted_jobs = sorted(
        job_history,
        key=lambda j: parse_date_to_year(j.get("start_date", "2000-01")) or 2000
    )
    
    previous_level = None
    
    for job in sorted_jobs:
        position = job.get("position", "").lower()
        
        # Infer career level from position title
        current_level = None
        if "junior" in position or "trainee" in position or "assistant" in position:
            current_level = "junior"
        elif "senior" in position:
            current_level = "senior"
        elif "lead" in position or "leiter" in position or "manager" in position or "head" in position:
            current_level = "lead"
        else:
            current_level = "mid"  # Default
        
        if previous_level and current_level:
            prev_rank = level_hierarchy.get(previous_level, 2)
            curr_rank = level_hierarchy.get(current_level, 2)
            
            # Regression: higher level → lower level
            if prev_rank > curr_rank and prev_rank - curr_rank > 1:
                issues.append(TimelineIssue(
                    severity="warning",
                    category="progression",
                    message=f"Career regression: {previous_level} → {current_level} at {job.get('company')}",
                    affected_periods=[],
                    suggested_fix=f"Adjust position title to maintain or improve career level"
                ))
        
        previous_level = current_level
    
    return issues


def check_age_consistency(
    persona_age: int,
    years_experience: int,
    education_history: List[Dict[str, Any]],
    job_history: List[Dict[str, Any]],
    career_level: str
) -> List[TimelineIssue]:
    """
    Check age/experience/career_level consistency.
    
    Args:
        persona_age: Persona's current age.
        years_experience: Years of work experience.
        education_history: List of education entries.
        job_history: List of job entries.
        career_level: Career level (junior, mid, senior, lead).
    
    Returns:
        List of age consistency issues.
    """
    issues = []
    current_year = datetime.now().year
    
    # Rule 1: Min age for career level
    min_ages = {
        "junior": 18,
        "mid": 20,
        "senior": 25,
        "lead": 30
    }
    min_age = min_ages.get(career_level, 18)
    
    if persona_age < min_age:
        issues.append(TimelineIssue(
            severity="error",
            category="age",
            message=f"Age {persona_age} too young for career level {career_level} (min: {min_age})",
            affected_periods=[],
            suggested_fix=f"Increase age to at least {min_age} or change career level"
        ))
    
    # Rule 2: Max years_experience for age
    max_years_exp = persona_age - 18
    if years_experience > max_years_exp:
        issues.append(TimelineIssue(
            severity="error",
            category="age",
            message=f"Years of experience ({years_experience}) exceeds maximum for age {persona_age} (max: {max_years_exp})",
            affected_periods=[],
            suggested_fix=f"Reduce years_experience to {max_years_exp} or increase age"
        ))
    
    # Rule 3: Education end + total job years ≈ current age - 18
    education_end = None
    if education_history:
        education_end = max([e.get("end_year", 0) for e in education_history])
    
    total_job_years = calculate_total_job_years(job_history)
    
    if education_end:
        expected_years = current_year - education_end
        discrepancy = abs(total_job_years - expected_years)
        
        if discrepancy > 2.0:  # Allow 2 years tolerance
            issues.append(TimelineIssue(
                severity="warning",
                category="age",
                message=f"Timeline discrepancy: Education ended {current_year - education_end} years ago, but job history shows {total_job_years:.1f} years",
                affected_periods=[],
                suggested_fix=f"Adjust job dates to match {expected_years:.1f} years of experience"
            ))
    
    return issues


def check_job_durations(job_history: List[Dict[str, Any]]) -> List[TimelineIssue]:
    """
    Check for realistic job durations (0.5-10 years per position).
    
    Args:
        job_history: List of job entries.
    
    Returns:
        List of duration issues.
    """
    issues = []
    current_year = datetime.now().year
    
    for job in job_history:
        start_date = job.get("start_date")
        end_date = job.get("end_date")
        is_current = job.get("is_current", False)
        
        if not start_date:
            continue
        
        start_year = parse_date_to_year(start_date)
        if not start_year:
            continue
        
        if is_current or not end_date:
            end_year = current_year
        else:
            end_year = parse_date_to_year(end_date)
            if not end_year:
                continue
        
        duration = end_year - start_year
        
        if duration < 0.5:
            issues.append(TimelineIssue(
                severity="warning",
                category="duration",
                message=f"Job at {job.get('company')} too short: {duration:.1f} years (min: 0.5)",
                affected_periods=[(start_year, end_year)],
                suggested_fix="Extend job duration to at least 0.5 years"
            ))
        elif duration > 10:
            issues.append(TimelineIssue(
                severity="info",
                category="duration",
                message=f"Job at {job.get('company')} very long: {duration:.1f} years (typical max: 10)",
                affected_periods=[(start_year, end_year)],
                suggested_fix="Consider splitting into multiple positions or adjusting dates"
            ))
    
    return issues


def auto_fix_overlaps(job_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Auto-fix overlapping job periods.
    
    Args:
        job_history: List of job entries.
    
    Returns:
        Fixed job history.
    """
    fixed = []
    current_year = datetime.now().year
    
    # Sort jobs by start date
    sorted_jobs = sorted(
        job_history,
        key=lambda j: parse_date_to_year(j.get("start_date", "2000-01")) or 2000
    )
    
    for i, job in enumerate(sorted_jobs):
        start = parse_date_to_year(job.get("start_date"))
        end = parse_date_to_year(job.get("end_date")) if not job.get("is_current") else current_year
        
        if i > 0:
            # Check overlap with previous job
            prev_job = sorted_jobs[i - 1]
            prev_start = parse_date_to_year(prev_job.get("start_date"))
            prev_end = parse_date_to_year(prev_job.get("end_date")) if not prev_job.get("is_current") else current_year
            
            if prev_end and start and prev_end >= start:
                # Fix: adjust previous job end to 1 year before current start
                fix_year = start - 1
                if fix_year < prev_start:
                    # Can't fix by adjusting previous, adjust current instead
                    fix_year = prev_end + 1
                    job["start_date"] = year_to_date_string(fix_year, 1)
                else:
                    prev_job["end_date"] = year_to_date_string(fix_year, 12)
                    prev_job["is_current"] = False
        
        fixed.append(job)
    
    return fixed


def insert_gap_filler(
    start_year: int,
    end_year: int,
    gap_type: str = "auto"
) -> Dict[str, Any]:
    """
    Insert a gap filler entry.
    
    Args:
        start_year: Gap start year.
        end_year: Gap end year.
        gap_type: Type of gap ("elternzeit", "sabbatical", "freelance", "auto").
    
    Returns:
        Gap filler job entry.
    """
    gap_fillers = {
        "elternzeit": {
            "de": "Elternzeit",
            "fr": "Congé parental",
            "it": "Congedo parentale"
        },
        "sabbatical": {
            "de": "Sabbatical / Weiterbildung",
            "fr": "Sabbatique / Formation continue",
            "it": "Sabbatico / Formazione continua"
        },
        "freelance": {
            "de": "Freelance Projekte",
            "fr": "Projets freelance",
            "it": "Progetti freelance"
        }
    }
    
    if gap_type == "auto":
        # Choose based on gap duration
        duration = end_year - start_year
        if duration >= 2:
            gap_type = "elternzeit"
        elif duration >= 1:
            gap_type = "sabbatical"
        else:
            gap_type = "freelance"
    
    filler_name = gap_fillers.get(gap_type, gap_fillers["freelance"])["de"]
    
    return {
        "company": filler_name,
        "position": filler_name,
        "location": "",
        "start_date": year_to_date_string(start_year, 1),
        "end_date": year_to_date_string(end_year - 1, 12),
        "is_current": False,
        "responsibilities": [],
        "technologies": [],
        "category": "gap_filler"
    }


def auto_fix_gaps(
    education_history: List[Dict[str, Any]],
    job_history: List[Dict[str, Any]],
    max_gap_months: int = 12
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Auto-fix gaps by inserting gap fillers.
    
    Args:
        education_history: List of education entries.
        job_history: List of job entries.
        max_gap_months: Maximum allowed gap before inserting filler.
    
    Returns:
        Tuple of (fixed_education_history, fixed_job_history).
    """
    fixed_jobs = []
    current_year = datetime.now().year
    
    # Get education end
    education_end = None
    if education_history:
        education_end = max([e.get("end_year", 0) for e in education_history])
    
    # Sort jobs by start date (excluding gap fillers)
    real_jobs = [j for j in job_history if j.get("category") != "gap_filler"]
    sorted_jobs = sorted(
        real_jobs,
        key=lambda j: parse_date_to_year(j.get("start_date", "2000-01")) or 2000
    )
    
    # Check gap between education and first job
    if education_end and sorted_jobs:
        first_job_start = parse_date_to_year(sorted_jobs[0].get("start_date"))
        if first_job_start:
            gap_years = first_job_start - education_end
            if gap_years > 0 and gap_years * 12 > max_gap_months:
                # Insert gap filler
                filler = insert_gap_filler(education_end, first_job_start)
                fixed_jobs.append(filler)
    
    # Check gaps between jobs
    for i, job in enumerate(sorted_jobs):
        fixed_jobs.append(job)
        
        if i < len(sorted_jobs) - 1:
            next_job = sorted_jobs[i + 1]
            
            end = parse_date_to_year(job.get("end_date")) if not job.get("is_current") else current_year
            start = parse_date_to_year(next_job.get("start_date"))
            
            if end and start:
                gap_years = start - end
                if gap_years > 0 and gap_years * 12 > max_gap_months:
                    # Insert gap filler (ensure no overlap)
                    filler_start = end + 1
                    filler_end = start - 1
                    if filler_start < filler_end:
                        filler = insert_gap_filler(filler_start, filler_end + 1)
                        fixed_jobs.append(filler)
    
    # Re-sort all jobs (including gap fillers) by start date
    fixed_jobs = sorted(
        fixed_jobs,
        key=lambda j: parse_date_to_year(j.get("start_date", "2000-01")) or 2000
    )
    
    return education_history, fixed_jobs


def adjust_dates_minor(
    education_history: List[Dict[str, Any]],
    job_history: List[Dict[str, Any]],
    persona_age: int,
    years_experience: int,
    tolerance_years: int = 1
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Adjust dates for minor inconsistencies (±tolerance_years).
    
    Args:
        education_history: List of education entries.
        job_history: List of job entries.
        persona_age: Persona's current age.
        years_experience: Years of work experience.
        tolerance_years: Tolerance in years for adjustments.
    
    Returns:
        Tuple of (adjusted_education_history, adjusted_job_history).
    """
    current_year = datetime.now().year
    
    # Adjust education end to align with work start
    if education_history and job_history:
        education_end = max([e.get("end_year", 0) for e in education_history])
        first_job_start = parse_date_to_year(sorted(job_history, key=lambda j: parse_date_to_year(j.get("start_date", "2000-01")) or 2000)[0].get("start_date"))
        
        if education_end and first_job_start:
            discrepancy = first_job_start - education_end
            if abs(discrepancy) <= tolerance_years:
                # Adjust education end to match first job start
                for entry in education_history:
                    if entry.get("end_year") == education_end:
                        entry["end_year"] = first_job_start
                        entry["start_year"] = entry["start_year"] + (first_job_start - education_end)
                        break
    
    # Adjust job dates to match total years_experience
    total_job_years = calculate_total_job_years(job_history)
    discrepancy = years_experience - total_job_years
    
    if abs(discrepancy) <= tolerance_years and job_history:
        # Adjust current job start date
        current_job = next((j for j in job_history if j.get("is_current")), None)
        if current_job:
            start = parse_date_to_year(current_job.get("start_date"))
            if start:
                # Adjust start to match years_experience
                new_start = current_year - int(years_experience)
                if abs(new_start - start) <= tolerance_years:
                    current_job["start_date"] = year_to_date_string(new_start, 1)
    
    return education_history, job_history


def validate_cv_timeline(
    persona: Dict[str, Any],
    education_history: List[Dict[str, Any]],
    job_history: List[Dict[str, Any]],
    auto_fix: bool = True,
    strict: bool = False
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[TimelineIssue]]:
    """
    Validate and fix CV timeline consistency.
    
    Args:
        persona: Persona dictionary with age, years_experience, career_level.
        education_history: List of education entries.
        job_history: List of job entries.
        auto_fix: Whether to automatically fix issues.
        strict: Whether to raise ValidationError on critical issues.
    
    Returns:
        Tuple of (validated_education_history, validated_job_history, issues).
    
    Raises:
        ValidationError: If strict=True and critical issues found.
    """
    issues = []
    
    persona_age = persona.get("age", 25)
    years_experience = persona.get("years_experience", 0)
    career_level = persona.get("career_level", "mid")
    
    # Run all validations
    issues.extend(check_job_overlaps(job_history))
    issues.extend(check_gaps(education_history, job_history))
    issues.extend(check_career_progression(job_history))
    issues.extend(check_age_consistency(persona_age, years_experience, education_history, job_history, career_level))
    issues.extend(check_job_durations(job_history))
    
    # Auto-fix if enabled
    fixed_education = education_history.copy()
    fixed_jobs = job_history.copy()
    
    if auto_fix:
        # Fix overlaps
        overlap_issues = [i for i in issues if i.category == "overlap" and i.severity == "error"]
        if overlap_issues:
            fixed_jobs = auto_fix_overlaps(fixed_jobs)
        
        # Fix gaps
        gap_issues = [i for i in issues if i.category == "gap"]
        if gap_issues:
            fixed_education, fixed_jobs = auto_fix_gaps(fixed_education, fixed_jobs)
        
        # Minor date adjustments
        fixed_education, fixed_jobs = adjust_dates_minor(
            fixed_education, fixed_jobs, persona_age, years_experience
        )
        
        # Re-validate after fixes
        issues_after_fix = []
        issues_after_fix.extend(check_job_overlaps(fixed_jobs))
        issues_after_fix.extend(check_gaps(fixed_education, fixed_jobs))
        
        # Only keep issues that weren't fixed
        critical_issues = [i for i in issues_after_fix if i.severity == "error"]
        if critical_issues:
            issues = [i for i in issues if i.severity == "error"] + critical_issues
        else:
            # Remove fixed issues
            issues = [i for i in issues if i.category not in ["overlap", "gap"] or i.severity != "error"]
    
    # Check for critical errors
    critical_errors = [i for i in issues if i.severity == "error"]
    
    if strict and critical_errors:
        raise ValidationError(
            f"CV timeline validation failed with {len(critical_errors)} critical errors",
            critical_errors
        )
    
    return fixed_education, fixed_jobs, issues


def get_timeline_summary(
    education_history: List[Dict[str, Any]],
    job_history: List[Dict[str, Any]],
    persona_age: int
) -> Dict[str, Any]:
    """
    Get summary of timeline for debugging.
    
    Args:
        education_history: List of education entries.
        job_history: List of job entries.
        persona_age: Persona's current age.
    
    Returns:
        Summary dictionary.
    """
    current_year = datetime.now().year
    
    education_end = None
    if education_history:
        education_end = max([e.get("end_year", 0) for e in education_history])
    
    total_job_years = calculate_total_job_years(job_history)
    
    first_job_start = None
    if job_history:
        sorted_jobs = sorted(
            job_history,
            key=lambda j: parse_date_to_year(j.get("start_date", "2000-01")) or 2000
        )
        first_job_start = parse_date_to_year(sorted_jobs[0].get("start_date"))
    
    return {
        "persona_age": persona_age,
        "education_end_year": education_end,
        "first_job_start_year": first_job_start,
        "total_job_years": round(total_job_years, 1),
        "years_since_education": current_year - education_end if education_end else None,
        "years_since_first_job": current_year - first_job_start if first_job_start else None,
        "num_education_entries": len(education_history),
        "num_job_entries": len(job_history)
    }

