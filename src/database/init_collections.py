"""
Initialize MongoDB collections in TARGET database only.

This script creates collections in the TARGET database (swiss_cv_generator):
- cantons: Canton information
- first_names: First name data
- last_names: Last name data
- companies: Company information
- occupation_skills: Skills associated with occupations

Note: The occupations collection is NOT created here.
It remains in the SOURCE database (CV_DATA.cv_berufsberatung) and is read directly from there.

Run: python src/database/init_collections.py
"""
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.mongodb_manager import get_db_manager
from pymongo.errors import OperationFailure


def init_cantons_collection(db_manager) -> None:
    """Initialize cantons collection with indexes in TARGET DB."""
    collection = db_manager.get_target_collection("cantons")
    
    # Create unique index on code
    collection.create_index([("code", 1)], unique=True, name="code_unique")
    
    # Create index on population
    collection.create_index([("population", 1)], name="population_idx")
    
    print("✅ cantons collection initialized in TARGET DB")
    print("   - Unique index: code")
    print("   - Index: population")
    print("   Schema: code, name_de/fr/it, population, workforce, language_de/fr/it, major_city, created_at")


def init_first_names_collection(db_manager) -> None:
    """Initialize first_names collection with indexes in TARGET DB."""
    collection = db_manager.get_target_collection("first_names")
    
    # Create unique compound index on name+language+gender
    collection.create_index(
        [("name", 1), ("language", 1), ("gender", 1)],
        unique=True,
        name="name_language_gender_unique"
    )
    
    # Create compound index on language and frequency
    collection.create_index([("language", 1), ("frequency", -1)], name="language_frequency_idx")
    
    print("✅ first_names collection initialized in TARGET DB")
    print("   - Unique compound index: name+language+gender")
    print("   - Compound index: language, frequency")
    print("   Schema: name, gender, language, frequency, origin")


def init_last_names_collection(db_manager) -> None:
    """Initialize last_names collection with indexes in TARGET DB."""
    collection = db_manager.get_target_collection("last_names")
    
    # Create unique compound index on name+language
    collection.create_index(
        [("name", 1), ("language", 1)],
        unique=True,
        name="name_language_unique"
    )
    
    # Create compound index on language and frequency
    collection.create_index([("language", 1), ("frequency", -1)], name="language_frequency_idx")
    
    print("✅ last_names collection initialized in TARGET DB")
    print("   - Unique compound index: name+language")
    print("   - Compound index: language, frequency")
    print("   Schema: name, language, frequency, origin")


def init_companies_collection(db_manager) -> None:
    """Initialize companies collection with indexes in TARGET DB."""
    collection = db_manager.get_target_collection("companies")
    
    # Create unique index on name
    collection.create_index([("name", 1)], unique=True, name="name_unique")
    
    # Create compound index on canton_code and industry
    collection.create_index(
        [("canton_code", 1), ("industry", 1)],
        name="canton_code_industry_idx"
    )
    
    print("✅ companies collection initialized in TARGET DB")
    print("   - Unique index: name")
    print("   - Compound index: canton_code+industry")
    print("   Schema: name, canton_code, industry, size_band, founded, is_real")


def init_occupation_skills_collection(db_manager) -> None:
    """Initialize occupation_skills collection with indexes in TARGET DB."""
    collection = db_manager.get_target_collection("occupation_skills")
    
    # Create index on job_id
    collection.create_index([("job_id", 1)], name="job_id_idx")
    
    # Create unique compound index on job_id+skill_name_de
    collection.create_index(
        [("job_id", 1), ("skill_name_de", 1)],
        unique=True,
        name="job_id_skill_name_de_unique"
    )
    
    print("✅ occupation_skills collection initialized in TARGET DB")
    print("   - Index: job_id")
    print("   - Unique compound index: job_id+skill_name_de")
    print("   Schema: job_id, skill_name_de/fr/it, skill_category, importance")


def main():
    """Main function to initialize all collections in TARGET DB."""
    print("=" * 60)
    print("Initializing MongoDB Collections in TARGET Database")
    print("=" * 60)
    print()
    
    try:
        # Get database manager
        db_manager = get_db_manager()
        
        # Connect to MongoDB
        print("Connecting to MongoDB...")
        db_manager.connect()
        print(f"✅ Connected to MongoDB")
        print(f"   Source DB: {db_manager.source_db.name} (read-only, unchanged)")
        print(f"   Target DB: {db_manager.target_db.name} (will be initialized)")
        print()
        print("⚠️  Note: occupations collection remains in SOURCE DB (CV_DATA.cv_berufsberatung)")
        print("   It will be read directly from there, not created in TARGET DB.")
        print()
        
        # Initialize all collections in TARGET DB
        init_cantons_collection(db_manager)
        print()
        
        init_first_names_collection(db_manager)
        print()
        
        init_last_names_collection(db_manager)
        print()
        
        init_companies_collection(db_manager)
        print()
        
        init_occupation_skills_collection(db_manager)
        print()
        
        print("=" * 60)
        print("✅ All collections initialized successfully in TARGET DB!")
        print("=" * 60)
        print()
        print("Collections created in TARGET DB (swiss_cv_generator):")
        print("  - cantons")
        print("  - first_names")
        print("  - last_names")
        print("  - companies")
        print("  - occupation_skills")
        print()
        print("Collections in SOURCE DB (CV_DATA) - unchanged:")
        print("  - cv_berufsberatung (read-only)")
        print()
        print("To verify, run in mongosh:")
        print("  use swiss_cv_generator")
        print("  show collections")
        print()
        print("To see indexes for a collection:")
        print("  db.cantons.getIndexes()")
        print("  db.first_names.getIndexes()")
        print("  # etc.")
        
    except OperationFailure as e:
        print(f"❌ MongoDB operation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Close connection
        try:
            db_manager.close()
        except:
            pass


if __name__ == "__main__":
    main()
