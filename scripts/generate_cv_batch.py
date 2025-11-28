"""
Batch CV Generation Script.

Generate large batches of CVs (100-1000+) with:
- Parallel processing
- Checkpoint system
- Statistical validation
- Quality metrics
- Comprehensive reporting

Run: python scripts/generate_cv_batch.py --count 1000 --parallel 4
"""
import sys
import os
import json
import time
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
import multiprocessing as mp
from multiprocessing import Pool, Manager
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generation.sampling import SamplingEngine
from src.generation.cv_assembler import generate_complete_cv, CVDocument
from src.generation.cv_timeline_validator import validate_cv_timeline
from src.generation.cv_quality_validator import validate_cv_quality, save_validation_report
from src.cli.main import export_cv_pdf, export_cv_docx, export_cv_json, filter_persona, get_age_group
from src.database.queries import get_occupation_by_id

console = Console()


@dataclass
class GenerationStats:
    """Statistics for batch generation."""
    total_generated: int = 0
    total_failed: int = 0
    total_retried: int = 0
    total_filtered: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # Demographic distribution
    age_groups: Dict[str, int] = field(default_factory=dict)
    genders: Dict[str, int] = field(default_factory=dict)
    industries: Dict[str, int] = field(default_factory=dict)
    career_levels: Dict[str, int] = field(default_factory=dict)
    languages: Dict[str, int] = field(default_factory=dict)
    cantons: Dict[str, int] = field(default_factory=dict)
    
    # Quality metrics
    quality_scores: List[float] = field(default_factory=list)
    validation_errors: int = 0
    validation_warnings: int = 0
    failed_validations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    generation_times: List[float] = field(default_factory=list)
    ai_api_calls: int = 0
    estimated_cost: float = 0.0
    
    # Career level by age group
    career_by_age: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "total_generated": self.total_generated,
            "total_failed": self.total_failed,
            "total_retried": self.total_retried,
            "total_filtered": self.total_filtered,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": (self.end_time - self.start_time) if self.end_time and self.start_time else 0,
            "demographics": {
                "age_groups": self.age_groups,
                "genders": self.genders,
                "industries": self.industries,
                "career_levels": self.career_levels,
                "languages": self.languages,
                "cantons": self.cantons,
                "career_by_age": dict(self.career_by_age)
            },
            "quality": {
                "avg_score": sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0,
                "min_score": min(self.quality_scores) if self.quality_scores else 0,
                "max_score": max(self.quality_scores) if self.quality_scores else 0,
                "validation_errors": self.validation_errors,
                "validation_warnings": self.validation_warnings,
                "failed_validations": self.failed_validations
            },
            "performance": {
                "avg_generation_time": sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0,
                "total_generation_time": sum(self.generation_times),
                "ai_api_calls": self.ai_api_calls,
                "estimated_cost": self.estimated_cost
            }
        }


@dataclass
class Checkpoint:
    """Checkpoint data for resuming generation."""
    count: int
    stats: GenerationStats
    generated_ids: List[str]
    timestamp: str


def generate_single_cv(
    args: Tuple[Dict[str, Any], int]
) -> Tuple[Optional[Dict[str, Any]], Optional[str], float]:
    """
    Generate a single CV (for multiprocessing).
    
    Args:
        args: Tuple of (config_dict, attempt_number).
    
    Returns:
        Tuple of (cv_data_dict, error_message, generation_time).
    """
    config, attempt = args
    start_time = time.time()
    
    try:
        # Initialize engine in worker process
        # Note: Each worker needs its own engine instance
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        from src.generation.sampling import SamplingEngine
        from src.generation.cv_assembler import generate_complete_cv
        from src.generation.cv_timeline_validator import validate_cv_timeline
        from src.generation.cv_quality_validator import validate_cv_quality
        from src.cli.main import filter_persona
        
        engine = SamplingEngine()
        
        # Sample persona
        persona = engine.sample_persona(
            preferred_canton=config.get("preferred_canton"),
            preferred_industry=config.get("preferred_industry")
        )
        
        # Apply filters
        if not filter_persona(
            persona,
            config.get("industry"),
            config.get("career_level"),
            config.get("age_group"),
            config.get("language")
        ):
            return None, "filtered", time.time() - start_time
        
        # Generate CV
        cv_doc = generate_complete_cv(persona)
        
        # Override language if specified
        if config.get("language"):
            cv_doc.language = config["language"]
        
        # Override portrait if disabled
        if not config.get("with_portrait", True):
            cv_doc.portrait_path = None
            cv_doc.portrait_base64 = None
        
        # Validate timeline
        if config.get("validate_timeline", True):
            validated_education, validated_jobs, _ = validate_cv_timeline(
                persona,
                cv_doc.education,
                cv_doc.jobs,
                auto_fix=True,
                strict=False
            )
            cv_doc.education = validated_education
            cv_doc.jobs = validated_jobs
        
        # Validate quality
        quality_score = 100.0
        if config.get("validate_quality", True):
            passed, quality_report = validate_cv_quality(
                cv_doc,
                min_score=config.get("min_quality_score", 80.0),
                save_report=False
            )
            
            quality_score = quality_report.score.overall
            
            if not passed:
                if attempt < config.get("max_retries", 3):
                    # Retry
                    return None, "quality_failed_retry", time.time() - start_time
                else:
                    return None, "quality_failed", time.time() - start_time
        
        generation_time = time.time() - start_time
        
        # Return CV data (serialize for multiprocessing)
        # Note: CVDocument needs to be serialized, so we convert to dict
        cv_data = {
            "cv_doc_dict": cv_doc.to_dict(),
            "persona": persona,
            "quality_score": quality_score
        }
        
        return cv_data, None, generation_time
        
    except Exception as e:
        return None, str(e), time.time() - start_time


def update_stats(stats: GenerationStats, cv_data: Dict[str, Any], generation_time: float):
    """Update statistics with new CV data."""
    # cv_data can be either dict with cv_doc_dict or with cv_doc
    if "cv_doc_dict" in cv_data:
        # Reconstruct from dict
        cv_doc_dict = cv_data["cv_doc_dict"]
        personal = cv_doc_dict.get("personal", {})
        professional = cv_doc_dict.get("professional", {})
        # Use dict directly for stats
        cv_doc_dict_for_stats = {
            "age": personal.get("age", 0),
            "gender": personal.get("gender", ""),
            "industry": professional.get("industry", ""),
            "career_level": professional.get("career_level", ""),
            "language": cv_doc_dict.get("metadata", {}).get("language", "de"),
            "canton": personal.get("canton", "")
        }
    else:
        # Direct cv_doc object
        cv_doc = cv_data["cv_doc"]
        cv_doc_dict_for_stats = {
            "age": cv_doc.age,
            "gender": cv_doc.gender,
            "industry": cv_doc.industry,
            "career_level": cv_doc.career_level,
            "language": cv_doc.language,
            "canton": cv_doc.canton
        }
    
    persona = cv_data.get("persona", {})
    
    stats.total_generated += 1
    stats.generation_times.append(generation_time)
    
    # Demographic distribution
    age = cv_doc_dict_for_stats.get("age", 0)
    age_grp = get_age_group(age)
    stats.age_groups[age_grp] = stats.age_groups.get(age_grp, 0) + 1
    stats.genders[cv_doc_dict_for_stats.get("gender", "")] = stats.genders.get(cv_doc_dict_for_stats.get("gender", ""), 0) + 1
    stats.industries[cv_doc_dict_for_stats.get("industry", "")] = stats.industries.get(cv_doc_dict_for_stats.get("industry", ""), 0) + 1
    stats.career_levels[cv_doc_dict_for_stats.get("career_level", "")] = stats.career_levels.get(cv_doc_dict_for_stats.get("career_level", ""), 0) + 1
    stats.languages[cv_doc_dict_for_stats.get("language", "de")] = stats.languages.get(cv_doc_dict_for_stats.get("language", "de"), 0) + 1
    stats.cantons[cv_doc_dict_for_stats.get("canton", "")] = stats.cantons.get(cv_doc_dict_for_stats.get("canton", ""), 0) + 1
    
    # Career level by age group
    stats.career_by_age[age_grp][cv_doc_dict_for_stats.get("career_level", "")] = stats.career_by_age[age_grp].get(cv_doc_dict_for_stats.get("career_level", ""), 0) + 1
    
    # Quality metrics
    quality_score = cv_data.get("quality_score", 100.0)
    stats.quality_scores.append(quality_score)


def save_checkpoint(checkpoint_path: Path, checkpoint: Checkpoint):
    """Save checkpoint to file."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(checkpoint_path: Path) -> Optional[Checkpoint]:
    """Load checkpoint from file."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    return None


def create_index_html(output_dir: Path, stats: GenerationStats) -> Path:
    """Create HTML index with CV previews."""
    index_path = output_dir / "index.html"
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Swiss CV Generator - Batch Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
        .stat-label {{ color: #7f8c8d; font-size: 14px; }}
        .distribution {{ background: white; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .distribution h3 {{ margin-top: 0; }}
        .bar {{ background: #3498db; height: 20px; margin: 5px 0; border-radius: 3px; display: flex; align-items: center; padding: 0 10px; color: white; }}
        table {{ width: 100%; border-collapse: collapse; background: white; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #34495e; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🇨🇭 Swiss CV Generator - Batch Results</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{stats.total_generated}</div>
            <div class="stat-label">Total Generated</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats.total_failed}</div>
            <div class="stat-label">Failed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{(sum(stats.quality_scores) / len(stats.quality_scores)) if stats.quality_scores else 0:.1f}</div>
            <div class="stat-label">Avg Quality Score</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{(sum(stats.generation_times) / len(stats.generation_times)) if stats.generation_times else 0:.2f}s</div>
            <div class="stat-label">Avg Generation Time</div>
        </div>
    </div>
    
    <div class="distribution">
        <h3>Age Group Distribution</h3>
        <p>Expected: 18-25: 7.6%, 26-40: 18.5%, 41-65: 31.0%</p>
        {_generate_distribution_bars(stats.age_groups, stats.total_generated)}
    </div>
    
    <div class="distribution">
        <h3>Gender Distribution</h3>
        <p>Expected: Male: 50.1%, Female: 49.9%</p>
        {_generate_distribution_bars(stats.genders, stats.total_generated)}
    </div>
    
    <div class="distribution">
        <h3>Industry Distribution</h3>
        {_generate_distribution_bars(stats.industries, stats.total_generated)}
    </div>
    
    <div class="distribution">
        <h3>Career Level Distribution</h3>
        {_generate_distribution_bars(stats.career_levels, stats.total_generated)}
    </div>
    
    <div class="distribution">
        <h3>Career Level by Age Group</h3>
        <table>
            <tr>
                <th>Age Group</th>
                <th>Junior</th>
                <th>Mid</th>
                <th>Senior</th>
                <th>Lead</th>
            </tr>
            {_generate_career_by_age_table(stats.career_by_age)}
        </table>
    </div>
</body>
</html>
"""
    
    index_path.write_text(html_content, encoding='utf-8')
    return index_path


def _generate_distribution_bars(distribution: Dict[str, int], total: int) -> str:
    """Generate HTML bars for distribution."""
    bars = []
    for key, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total * 100) if total > 0 else 0
        bars.append(f'<div class="bar" style="width: {percentage}%">{key}: {count} ({percentage:.1f}%)</div>')
    return '\n'.join(bars)


def _generate_career_by_age_table(career_by_age: Dict[str, Dict[str, int]]) -> str:
    """Generate HTML table for career level by age group."""
    rows = []
    for age_group in ["18-25", "26-40", "41-65"]:
        if age_group in career_by_age:
            data = career_by_age[age_group]
            rows.append(f"""
            <tr>
                <td>{age_group}</td>
                <td>{data.get('junior', 0)}</td>
                <td>{data.get('mid', 0)}</td>
                <td>{data.get('senior', 0)}</td>
                <td>{data.get('lead', 0)}</td>
            </tr>
            """)
    return '\n'.join(rows)


@click.command()
@click.option('--count', '-n', default=100, type=int, help='Total CVs to generate')
@click.option('--parallel', '-p', default=4, type=int, help='Number of parallel workers')
@click.option('--checkpoint-every', default=50, type=int, help='Save checkpoint every N CVs')
@click.option('--min-quality-score', default=80.0, type=float, help='Minimum quality score to accept')
@click.option('--max-retries', default=3, type=int, help='Max retries for failed validations')
@click.option('--output-format', default='pdf', type=click.Choice(['pdf', 'docx', 'both']), help='Output format')
@click.option('--create-index', default=True, is_flag=True, help='Generate HTML index')
@click.option('--output-dir', '-o', default='output/batch', type=click.Path(), help='Output directory')
@click.option('--resume', is_flag=True, help='Resume from checkpoint')
@click.option('--industry', default=None, help='Filter by industry')
@click.option('--language', default='de', type=click.Choice(['de', 'fr', 'it']), help='Language')
def generate_batch(
    count: int,
    parallel: int,
    checkpoint_every: int,
    min_quality_score: float,
    max_retries: int,
    output_format: str,
    create_index: bool,
    output_dir: str,
    resume: bool,
    industry: Optional[str],
    language: str
):
    """
    Generate large batch of CVs with parallel processing and statistics.
    
    Examples:
    
    \b
        # Generate 1000 CVs with 4 parallel workers
        python scripts/generate_cv_batch.py --count 1000 --parallel 4
    
    \b
        # Generate with checkpoint every 100 CVs
        python scripts/generate_cv_batch.py --count 500 --checkpoint-every 100
    
    \b
        # Resume from checkpoint
        python scripts/generate_cv_batch.py --count 1000 --resume
    """
    console.print(Panel.fit("[bold green]🇨🇭 Swiss CV Generator - Batch Mode[/bold green]", border_style="green"))
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = output_path / "checkpoint.pkl"
    
    # Initialize stats
    if resume and checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            stats = checkpoint.stats
            generated_ids = set(checkpoint.generated_ids)
            start_count = checkpoint.count
            console.print(f"[yellow]Resuming from checkpoint: {start_count} CVs already generated[/yellow]")
        else:
            stats = GenerationStats()
            generated_ids = set()
            start_count = 0
    else:
        stats = GenerationStats()
        generated_ids = set()
        start_count = 0
    
    stats.start_time = stats.start_time or time.time()
    
    # Create output structure
    language_dir = output_path / language
    if industry:
        industry_dir = language_dir / industry
    else:
        industry_dir = language_dir / "all"
    
    industry_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration for workers
    config = {
        "industry": industry,
        "language": language,
        "preferred_canton": None,
        "preferred_industry": industry,
        "career_level": None,
        "age_group": None,
        "with_portrait": True,
        "validate_timeline": True,
        "validate_quality": True,
        "min_quality_score": min_quality_score,
        "max_retries": max_retries
    }
    
    # Progress tracking
    remaining = count - start_count
    total_attempts = 0
    max_total_attempts = remaining * (max_retries + 1) * 2  # Safety limit
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        TimeRemainingColumn()
    ) as progress:
        task = progress.add_task(f"[cyan]Generating CVs...", total=remaining)
        
        # Generate CVs
        pool = Pool(processes=parallel)
        
        try:
            while stats.total_generated < count and total_attempts < max_total_attempts:
                # Prepare batch of tasks
                batch_size = min(parallel * 2, count - stats.total_generated)
                tasks = [(config, 0) for _ in range(batch_size)]
                
                # Generate batch
                try:
                    results = pool.map(generate_single_cv, tasks)
                except Exception as e:
                    console.print(f"[red]Error in batch generation: {e}[/red]")
                    if verbose:
                        import traceback
                        console.print(traceback.format_exc())
                    break
                
                for cv_data, error, gen_time in results:
                    total_attempts += 1
                    
                    if error:
                        if error == "filtered":
                            stats.total_filtered += 1
                        elif error == "quality_failed_retry":
                            stats.total_retried += 1
                        elif error == "quality_failed":
                            stats.total_failed += 1
                            stats.failed_validations.append({"error": error, "timestamp": datetime.now().isoformat()})
                        else:
                            stats.total_failed += 1
                            stats.failed_validations.append({"error": error, "timestamp": datetime.now().isoformat()})
                        continue
                    
                    if cv_data:
                        # Reconstruct CVDocument from dict
                        from src.generation.cv_assembler import CVDocument
                        cv_doc_dict = cv_data["cv_doc_dict"]
                        personal = cv_doc_dict.get("personal", {})
                        professional = cv_doc_dict.get("professional", {})
                        content = cv_doc_dict.get("content", {})
                        metadata = cv_doc_dict.get("metadata", {})
                        
                        cv_doc = CVDocument(
                            first_name=personal.get("first_name", ""),
                            last_name=personal.get("last_name", ""),
                            full_name=personal.get("full_name", ""),
                            age=personal.get("age", 0),
                            gender=personal.get("gender", ""),
                            canton=personal.get("canton", ""),
                            city=personal.get("city"),
                            email=personal.get("email", ""),
                            phone=personal.get("phone", ""),
                            address=personal.get("address"),
                            portrait_path=personal.get("portrait_path"),
                            portrait_base64=personal.get("portrait_base64"),
                            current_title=professional.get("current_title", ""),
                            industry=professional.get("industry", ""),
                            career_level=professional.get("career_level", ""),
                            years_experience=professional.get("years_experience", 0),
                            summary=content.get("summary", ""),
                            education=content.get("education", []),
                            jobs=content.get("jobs", []),
                            skills=content.get("skills", {}),
                            additional_education=content.get("additional_education", []),
                            hobbies=content.get("hobbies", []),
                            language=metadata.get("language", "de"),
                            created_at=metadata.get("created_at", "")
                        )
                        
                        # Generate filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        job_id_str = str(cv_data["persona"].get("job_id", "unknown"))
                        filename_base = f"{cv_doc.last_name}_{cv_doc.first_name}_{job_id_str}_{timestamp}"
                        cv_id = f"{cv_doc.last_name}_{cv_doc.first_name}_{job_id_str}"
                        
                        if cv_id in generated_ids:
                            continue
                        
                        generated_ids.add(cv_id)
                        
                        # Export CV - JSON is the main output, PDF/DOCX are optional
                        export_success = False
                        
                        # Export metadata (always, this is the main output)
                        json_path = industry_dir / f"{filename_base}.json"
                        try:
                            export_cv_json(cv_doc, json_path)
                            export_success = True
                        except Exception as json_error:
                            stats.total_failed += 1
                            stats.failed_validations.append({
                                "error": f"JSON export failed: {str(json_error)}",
                                "cv_id": cv_id,
                                "timestamp": datetime.now().isoformat()
                            })
                            continue
                        
                        # Try PDF export (optional - failures don't count as overall failure)
                        if export_success and output_format in ('pdf', 'both'):
                            pdf_path = industry_dir / f"{filename_base}.pdf"
                            try:
                                export_cv_pdf(cv_doc, pdf_path)
                            except Exception:
                                # PDF failed, but JSON succeeded, so continue
                                pass
                        
                        # Try DOCX export (optional - failures don't count as overall failure)
                        if export_success and output_format in ('docx', 'both'):
                            docx_path = industry_dir / f"{filename_base}.docx"
                            try:
                                export_cv_docx(cv_doc, docx_path)
                            except Exception:
                                # DOCX failed, but JSON succeeded, so continue
                                pass
                        
                        if export_success:
                            # Update stats
                            update_stats(stats, cv_data, gen_time)
                            
                            # Update progress
                            current_name = f"{cv_doc.first_name} {cv_doc.last_name}"
                            progress.update(
                                task,
                                description=f"[cyan]Generated: {current_name} ({cv_doc.current_title[:30]}...)",
                                advance=1
                            )
                            
                            # Save checkpoint
                            if stats.total_generated % checkpoint_every == 0:
                                checkpoint = Checkpoint(
                                    count=stats.total_generated,
                                    stats=stats,
                                    generated_ids=list(generated_ids),
                                    timestamp=datetime.now().isoformat()
                                )
                                save_checkpoint(checkpoint_path, checkpoint)
        
        finally:
            pool.close()
            pool.join()
    
    stats.end_time = time.time()
    
    # Generate summary report
    console.print()
    console.print(Panel.fit("[bold blue]Generation Complete[/bold blue]", border_style="blue"))
    
    # Statistics table
    duration = stats.end_time - stats.start_time
    cvs_per_minute = (stats.total_generated / duration * 60) if duration > 0 else 0
    
    table = Table(title="Generation Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Generated", str(stats.total_generated))
    table.add_row("Total Failed", str(stats.total_failed))
    table.add_row("Total Retried", str(stats.total_retried))
    table.add_row("Total Filtered", str(stats.total_filtered))
    table.add_row("Duration", f"{duration:.1f}s ({duration/60:.1f}m)")
    table.add_row("Speed", f"{cvs_per_minute:.1f} CVs/minute")
    table.add_row("Avg Generation Time", f"{sum(stats.generation_times)/len(stats.generation_times):.2f}s" if stats.generation_times else "N/A")
    
    if stats.quality_scores:
        table.add_row("Avg Quality Score", f"{sum(stats.quality_scores)/len(stats.quality_scores):.1f}/100")
        table.add_row("Min Quality Score", f"{min(stats.quality_scores):.1f}/100")
        table.add_row("Max Quality Score", f"{max(stats.quality_scores):.1f}/100")
    
    console.print(table)
    
    # Demographic distribution
    console.print("\n[bold yellow]Demographic Distribution:[/bold yellow]")
    
    # Age groups
    age_table = Table(title="Age Groups", show_header=True)
    age_table.add_column("Age Group", style="cyan")
    age_table.add_column("Count", style="green")
    age_table.add_column("Percentage", style="yellow")
    age_table.add_column("Expected", style="magenta")
    
    expected_age = {"18-25": 7.6, "26-40": 18.5, "41-65": 31.0}
    for age_grp in ["18-25", "26-40", "41-65"]:
        count = stats.age_groups.get(age_grp, 0)
        pct = (count / stats.total_generated * 100) if stats.total_generated > 0 else 0
        expected = expected_age.get(age_grp, 0)
        age_table.add_row(age_grp, str(count), f"{pct:.1f}%", f"{expected}%")
    
    console.print(age_table)
    
    # Gender
    gender_table = Table(title="Gender Distribution", show_header=True)
    gender_table.add_column("Gender", style="cyan")
    gender_table.add_column("Count", style="green")
    gender_table.add_column("Percentage", style="yellow")
    gender_table.add_column("Expected", style="magenta")
    
    expected_gender = {"male": 50.1, "female": 49.9}
    for gender in ["male", "female"]:
        count = stats.genders.get(gender, 0)
        pct = (count / stats.total_generated * 100) if stats.total_generated > 0 else 0
        expected = expected_gender.get(gender, 0)
        gender_table.add_row(gender, str(count), f"{pct:.1f}%", f"{expected}%")
    
    console.print(gender_table)
    
    # Save full report
    report_path = output_path / "generation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats.to_dict(), f, indent=2, ensure_ascii=False)
    
    console.print(f"\n[green]✅ Full report saved to: {report_path}[/green]")
    
    # Create HTML index
    if create_index and stats.total_generated > 0:
        try:
            index_path = create_index_html(output_path, stats)
            console.print(f"[green]✅ HTML index created: {index_path}[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not create HTML index: {e}[/yellow]")
    
    console.print(f"\n[bold green]✅ Batch generation complete![/bold green]")
    console.print(f"[green]CVs saved to: {industry_dir}[/green]")


if __name__ == '__main__':
    generate_batch()

