"""
OccupationLoader - Loads and indexes occupation data from processed JSON file.
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


class OccupationLoader:
    """
    Loads and indexes occupation data from data/processed/occupations.json.
    
    Indexes occupations by industry for fast lookup.
    """
    
    def __init__(self, data_file: Optional[Path] = None):
        """
        Initialize the loader and load occupation data.
        
        Args:
            data_file: Optional path to occupations.json file.
                      Defaults to data/processed/occupations.json relative to project root.
        
        Raises:
            FileNotFoundError: If the occupations file doesn't exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        if data_file is None:
            # Get project root (assuming this file is in src/data/)
            project_root = Path(__file__).parent.parent.parent
            data_file = project_root / "data" / "processed" / "occupations.json"
        
        self.data_file = Path(data_file)
        
        if not self.data_file.exists():
            raise FileNotFoundError(
                f"Occupations file not found: {self.data_file}. "
                f"Please run scripts/process_occupations.py first."
            )
        
        # Load occupations
        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                self._occupations = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in {self.data_file}: {e.msg}",
                e.doc,
                e.pos
            )
        
        # Index by industry
        self._by_industry: Dict[str, List[Dict]] = defaultdict(list)
        self._by_id: Dict[str, Dict] = {}
        
        for occ in self._occupations:
            industry = occ.get("industry", "other")
            self._by_industry[industry].append(occ)
            
            occ_id = occ.get("id")
            if occ_id:
                self._by_id[occ_id] = occ
    
    def get_all(self) -> List[Dict]:
        """
        Get all occupations.
        
        Returns:
            List of all occupation dictionaries.
        """
        return self._occupations.copy()
    
    def get_by_industry(self, industry: str) -> List[Dict]:
        """
        Get all occupations for a specific industry.
        
        Args:
            industry: Industry name (e.g., 'technology', 'finance', 'healthcare', 
                     'construction', 'other').
        
        Returns:
            List of occupation dictionaries for the specified industry.
        """
        return self._by_industry.get(industry, []).copy()
    
    def sample_random(self, industry: Optional[str] = None) -> Optional[Dict]:
        """
        Sample a random occupation.
        
        Args:
            industry: Optional industry filter. If provided, samples only from
                     that industry. If None, samples from all occupations.
        
        Returns:
            Random occupation dictionary, or None if no occupations match.
        """
        if industry:
            occupations = self.get_by_industry(industry)
        else:
            occupations = self.get_all()
        
        if not occupations:
            return None
        
        return random.choice(occupations)
    
    def get_by_id(self, occ_id: str) -> Optional[Dict]:
        """
        Get an occupation by its ID.
        
        Args:
            occ_id: Occupation ID.
        
        Returns:
            Occupation dictionary if found, None otherwise.
        """
        return self._by_id.get(occ_id)


# Singleton instance
_occupation_loader_instance: Optional[OccupationLoader] = None


def get_occupation_loader(data_file: Optional[Path] = None) -> OccupationLoader:
    """
    Get the singleton OccupationLoader instance.
    
    Args:
        data_file: Optional path to occupations.json file.
                  Only used on first call. Subsequent calls ignore this parameter.
    
    Returns:
        The singleton OccupationLoader instance.
    """
    global _occupation_loader_instance
    
    if _occupation_loader_instance is None:
        _occupation_loader_instance = OccupationLoader(data_file)
    
    return _occupation_loader_instance

