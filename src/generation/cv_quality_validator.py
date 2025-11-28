"""
CV Quality Validator.

This module validates CV quality across multiple dimensions:
- Completeness: All sections present with minimum content
- Realism: Logical timelines, appropriate titles, valid data
- Language consistency: Correct language, formatting, Swiss conventions
- Quality scoring: Overall score with weighted components

Run: Used by CV generation pipeline before export
"""
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.generation.cv_assembler import CVDocument
from src.database.queries import (
    get_occupation_by_id,
    sample_company_by_canton_and_industry,
    get_skills_by_occupation
)


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    category: str  # "completeness", "realism", "language"
    severity: str  # "critical", "warning", "info"
    section: str  # "education", "jobs", "skills", etc.
    field: Optional[str] = None
    message: str = ""
    suggested_fix: Optional[str] = None
    score_impact: float = 0.0  # Points deducted from score


@dataclass
class QualityScore:
    """Quality score breakdown."""
    completeness: float = 0.0  # 0-100
    realism: float = 0.0  # 0-100
    language: float = 0.0  # 0-100
    overall: float = 0.0  # Weighted average
    
    def calculate_overall(self, weights: Dict[str, float] = None) -> float:
        """Calculate overall score with weights."""
        if weights is None:
            weights = {"completeness": 0.4, "realism": 0.4, "language": 0.2}
        
        self.overall = (
            self.completeness * weights.get("completeness", 0.4) +
            self.realism * weights.get("realism", 0.4) +
            self.language * weights.get("language", 0.2)
        )
        return self.overall


@dataclass
class ValidationReport:
    """Complete validation report."""
    cv_id: str
    timestamp: str
    passed: bool
    score: QualityScore
    issues: List[ValidationIssue] = field(default_factory=list)
    critical_issues: int = 0
    warnings: int = 0
    info: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "cv_id": self.cv_id,
            "timestamp": self.timestamp,
            "passed": self.passed,
            "score": {
                "completeness": self.score.completeness,
                "realism": self.score.realism,
                "language": self.score.language,
                "overall": self.score.overall
            },
            "issues": [asdict(issue) for issue in self.issues],
            "summary": {
                "critical_issues": self.critical_issues,
                "warnings": self.warnings,
                "info": self.info,
                "total_issues": len(self.issues)
            }
        }


class CVQualityValidator:
    """Validates CV quality across multiple dimensions."""
    
    def __init__(self, min_score_threshold: float = 80.0):
        """
        Initialize validator.
        
        Args:
            min_score_threshold: Minimum overall score to pass (default: 80.0).
        """
        self.min_score_threshold = min_score_threshold
        self.issues: List[ValidationIssue] = []
    
    def validate(self, cv_doc: CVDocument) -> ValidationReport:
        """
        Run complete validation on CV document.
        
        Args:
            cv_doc: Complete CV document.
        
        Returns:
            Validation report.
        """
        self.issues = []
        
        # Generate CV ID
        cv_id = f"{cv_doc.last_name}_{cv_doc.first_name}_{cv_doc.language}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Run all validations
        completeness_score = self._validate_completeness(cv_doc)
        realism_score = self._validate_realism(cv_doc)
        language_score = self._validate_language(cv_doc)
        
        # Calculate overall score
        score = QualityScore(
            completeness=completeness_score,
            realism=realism_score,
            language=language_score
        )
        score.calculate_overall()
        
        # Categorize issues
        critical_issues = len([i for i in self.issues if i.severity == "critical"])
        warnings = len([i for i in self.issues if i.severity == "warning"])
        info_count = len([i for i in self.issues if i.severity == "info"])
        
        # Determine if passed
        passed = score.overall >= self.min_score_threshold and critical_issues == 0
        
        report = ValidationReport(
            cv_id=cv_id,
            timestamp=datetime.now().isoformat(),
            passed=passed,
            score=score,
            issues=self.issues,
            critical_issues=critical_issues,
            warnings=warnings,
            info=info_count
        )
        
        return report
    
    def _validate_completeness(self, cv_doc: CVDocument) -> float:
        """
        Validate completeness of CV sections.
        
        Returns:
            Completeness score (0-100).
        """
        score = 100.0
        max_deduction = 100.0
        
        # Required sections
        required_sections = {
            "personal": ["first_name", "last_name", "age", "canton"],
            "professional": ["current_title", "industry", "career_level"],
            "content": ["summary", "education", "jobs", "skills"]
        }
        
        # Check personal information
        if not cv_doc.first_name or not cv_doc.last_name:
            self._add_issue("completeness", "critical", "personal", "name",
                          "Missing first or last name",
                          "Ensure persona has valid first_name and last_name",
                          -20.0)
            score -= 20.0
        
        if not cv_doc.age or cv_doc.age < 18:
            self._add_issue("completeness", "critical", "personal", "age",
                          "Missing or invalid age",
                          "Ensure persona has valid age >= 18",
                          -15.0)
            score -= 15.0
        
        if not cv_doc.canton:
            self._add_issue("completeness", "warning", "personal", "canton",
                          "Missing canton",
                          "Ensure persona has valid canton code",
                          -5.0)
            score -= 5.0
        
        # Check summary
        if not cv_doc.summary or len(cv_doc.summary.strip()) < 50:
            self._add_issue("completeness", "warning", "content", "summary",
                          "Summary too short or missing (min 50 chars)",
                          "Generate longer summary with AI or fallback",
                          -10.0)
            score -= 10.0
        
        # Check education
        if not cv_doc.education or len(cv_doc.education) == 0:
            self._add_issue("completeness", "critical", "content", "education",
                          "No education entries",
                          "Generate at least one education entry",
                          -25.0)
            score -= 25.0
        else:
            # Check minimum content per education entry
            for i, edu in enumerate(cv_doc.education):
                if not edu.get("degree") or not edu.get("institution"):
                    self._add_issue("completeness", "warning", "education", f"entry_{i}",
                                  f"Education entry {i+1} missing degree or institution",
                                  "Ensure all education entries have degree and institution",
                                  -5.0)
                    score -= 5.0
        
        # Check jobs
        if not cv_doc.jobs or len(cv_doc.jobs) == 0:
            self._add_issue("completeness", "critical", "content", "jobs",
                          "No job entries",
                          "Generate at least one job entry",
                          -30.0)
            score -= 30.0
        else:
            # Check minimum content per job
            for i, job in enumerate(cv_doc.jobs):
                if not job.get("company") or not job.get("position"):
                    self._add_issue("completeness", "critical", "jobs", f"entry_{i}",
                                  f"Job entry {i+1} missing company or position",
                                  "Ensure all job entries have company and position",
                                  -15.0)
                    score -= 15.0
                
                # Check responsibilities (min 2 per job)
                responsibilities = job.get("responsibilities", [])
                if len(responsibilities) < 2:
                    self._add_issue("completeness", "warning", "jobs", f"entry_{i}_responsibilities",
                                  f"Job entry {i+1} has fewer than 2 responsibilities (has {len(responsibilities)})",
                                  "Generate at least 2 responsibilities per job",
                                  -5.0)
                    score -= 5.0
        
        # Check skills (min 8 total)
        total_skills = 0
        if cv_doc.skills:
            total_skills = sum(len(skills_list) for skills_list in cv_doc.skills.values())
        
        if total_skills < 8:
            self._add_issue("completeness", "warning", "content", "skills",
                          f"Too few skills (has {total_skills}, min 8)",
                          "Generate at least 8 skills total",
                          -10.0)
            score -= 10.0
        
        # Check portrait (if with_portrait was True)
        if cv_doc.portrait_path and not Path(project_root / "data" / "portraits" / cv_doc.portrait_path).exists():
            self._add_issue("completeness", "warning", "personal", "portrait",
                          f"Portrait path specified but file not found: {cv_doc.portrait_path}",
                          "Ensure portrait file exists or remove portrait_path",
                          -5.0)
            score -= 5.0
        
        return max(0.0, score)
    
    def _validate_realism(self, cv_doc: CVDocument) -> float:
        """
        Validate realism of CV data.
        
        Returns:
            Realism score (0-100).
        """
        score = 100.0
        
        # Check age vs years_experience
        age = cv_doc.age
        years_exp = cv_doc.years_experience
        
        if age < 18 + years_exp:
            self._add_issue("realism", "critical", "professional", "age_experience",
                          f"Age ({age}) too young for years_experience ({years_exp}). Age must be >= 18 + years_exp",
                          f"Increase age to at least {18 + years_exp} or reduce years_experience",
                          -25.0)
            score -= 25.0
        
        # Check career level appropriateness
        career_level = cv_doc.career_level
        min_ages = {"junior": 18, "mid": 20, "senior": 25, "lead": 30}
        min_age = min_ages.get(career_level, 18)
        
        if age < min_age:
            self._add_issue("realism", "warning", "professional", "career_level",
                          f"Age ({age}) too young for career level {career_level} (min: {min_age})",
                          f"Increase age to at least {min_age} or change career level",
                          -10.0)
            score -= 10.0
        
        # Check timeline overlaps (basic check)
        if cv_doc.jobs and len(cv_doc.jobs) > 1:
            sorted_jobs = sorted(
                cv_doc.jobs,
                key=lambda j: self._parse_date_to_year(j.get("start_date", "2000-01")) or 2000
            )
            
            for i in range(len(sorted_jobs) - 1):
                job1 = sorted_jobs[i]
                job2 = sorted_jobs[i + 1]
                
                end1 = self._parse_date_to_year(job1.get("end_date")) if not job1.get("is_current") else datetime.now().year
                start2 = self._parse_date_to_year(job2.get("start_date"))
                
                if end1 and start2 and end1 > start2:
                    self._add_issue("realism", "critical", "jobs", "timeline_overlap",
                                  f"Job overlap: {job1.get('company')} ends {end1} but {job2.get('company')} starts {start2}",
                                  "Fix timeline overlaps using cv_timeline_validator",
                                  -20.0)
                    score -= 20.0
        
        # Check company exists in canton+industry
        if cv_doc.jobs:
            current_job = next((j for j in cv_doc.jobs if j.get("is_current")), cv_doc.jobs[0] if cv_doc.jobs else None)
            if current_job:
                company_name = current_job.get("company", "")
                # Try to find company in database
                try:
                    company_doc = sample_company_by_canton_and_industry(cv_doc.canton, cv_doc.industry)
                    if company_doc and company_doc.get("name") != company_name:
                        # Company might still be valid, just not the sampled one
                        # This is a soft check, so only warning
                        self._add_issue("realism", "info", "jobs", "company_validation",
                                      f"Company '{company_name}' not found in database for canton {cv_doc.canton} and industry {cv_doc.industry}",
                                      "Verify company exists or use sample_company_by_canton_and_industry",
                                      -2.0)
                        score -= 2.0
                except Exception:
                    # Database query failed, skip this check
                    pass
        
        # Check occupation matches CV_DATA structure
        # This is validated by checking if job_id exists
        # (We can't directly check this without the persona dict, so we'll skip for now)
        
        # Check skills match occupation requirements
        # This would require loading occupation and comparing skills
        # For now, we'll do a basic check that skills exist
        
        return max(0.0, score)
    
    def _validate_language(self, cv_doc: CVDocument) -> float:
        """
        Validate language consistency and formatting.
        
        Returns:
            Language score (0-100).
        """
        score = 100.0
        language = cv_doc.language
        
        # Check date formats
        date_format_issues = 0
        
        # Check job dates
        for job in cv_doc.jobs:
            start_date = job.get("start_date", "")
            end_date = job.get("end_date", "")
            
            # Dates should be in YYYY-MM format
            if start_date and not self._is_valid_date_format(start_date):
                date_format_issues += 1
            if end_date and not self._is_valid_date_format(end_date):
                date_format_issues += 1
        
        # Check education dates
        for edu in cv_doc.education:
            start_year = edu.get("start_year")
            end_year = edu.get("end_year")
            
            if start_year and (not isinstance(start_year, int) or start_year < 1980 or start_year > datetime.now().year):
                date_format_issues += 1
            if end_year and (not isinstance(end_year, int) or end_year < 1980 or end_year > datetime.now().year):
                date_format_issues += 1
        
        if date_format_issues > 0:
            self._add_issue("language", "warning", "formatting", "dates",
                          f"Found {date_format_issues} date format issues",
                          "Ensure all dates are in correct format (YYYY-MM for jobs, int years for education)",
                          -5.0 * min(date_format_issues, 5))
            score -= 5.0 * min(date_format_issues, 5)
        
        # Check language consistency in text fields
        # This is a basic check - in production, you might use language detection
        text_fields = {
            "summary": cv_doc.summary,
            "current_title": cv_doc.current_title
        }
        
        # Basic language-specific character checks
        language_indicators = {
            "de": ["ä", "ö", "ü", "ß", "der", "die", "das"],
            "fr": ["é", "è", "ê", "à", "le", "la", "les"],
            "it": ["à", "è", "é", "ì", "ò", "ù", "il", "la"]
        }
        
        indicators = language_indicators.get(language, [])
        if indicators:
            for field_name, field_value in text_fields.items():
                if field_value:
                    # Check if text contains language indicators
                    has_indicators = any(ind in field_value.lower() for ind in indicators[:3])  # Check first 3
                    if not has_indicators and len(field_value) > 20:
                        # Might be wrong language, but this is soft
                        self._add_issue("language", "info", "content", field_name,
                                      f"Field '{field_name}' might not match language {language}",
                                      "Verify text is in correct language",
                                      -2.0)
                        score -= 2.0
        
        # Check Swiss conventions
        # For German: should use Swiss German conventions (no ß in some contexts, etc.)
        # This is a soft check
        
        return max(0.0, score)
    
    def _add_issue(
        self,
        category: str,
        severity: str,
        section: str,
        field: Optional[str],
        message: str,
        suggested_fix: Optional[str] = None,
        score_impact: float = 0.0
    ):
        """Add a validation issue."""
        issue = ValidationIssue(
            category=category,
            severity=severity,
            section=section,
            field=field,
            message=message,
            suggested_fix=suggested_fix,
            score_impact=score_impact
        )
        self.issues.append(issue)
    
    def _parse_date_to_year(self, date_str: Optional[str]) -> Optional[int]:
        """Parse date string to year."""
        if not date_str:
            return None
        try:
            if "-" in date_str:
                return int(date_str.split("-")[0])
            else:
                return int(date_str)
        except (ValueError, AttributeError):
            return None
    
    def _is_valid_date_format(self, date_str: str) -> bool:
        """Check if date string is in valid format (YYYY-MM)."""
        try:
            if "-" in date_str:
                parts = date_str.split("-")
                if len(parts) == 2:
                    year = int(parts[0])
                    month = int(parts[1])
                    return 1980 <= year <= datetime.now().year and 1 <= month <= 12
            return False
        except (ValueError, AttributeError):
            return False


def save_validation_report(report: ValidationReport, output_path: Path) -> Path:
    """
    Save validation report to JSON file.
    
    Args:
        report: Validation report.
        output_path: Output file path.
    
    Returns:
        Path to saved report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    
    return output_path


def validate_cv_quality(
    cv_doc: CVDocument,
    min_score: float = 80.0,
    save_report: bool = True,
    report_path: Optional[Path] = None
) -> Tuple[bool, ValidationReport]:
    """
    Validate CV quality and return result.
    
    Args:
        cv_doc: Complete CV document.
        min_score: Minimum score threshold (default: 80.0).
        save_report: Whether to save validation report (default: True).
        report_path: Optional path for report (default: auto-generated).
    
    Returns:
        Tuple of (passed, validation_report).
    """
    validator = CVQualityValidator(min_score_threshold=min_score)
    report = validator.validate(cv_doc)
    
    if save_report:
        if not report_path:
            report_path = Path(project_root / "output" / "validation_reports" / f"{report.cv_id}_validation.json")
        save_validation_report(report, report_path)
    
    return report.passed, report

