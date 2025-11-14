"""Utility functions for generating and mapping faculty IDs."""

import re
from typing import Optional, Dict
from src.utils.name_matcher import names_match


def generate_faculty_id(name: str) -> str:
    """
    Generate a faculty_id from a name using pattern: first initial + last name (lowercase).

    Examples:
        "Charles Nicholson" -> "cnicholson"
        "Jie Cao" -> "jcao"
        "Le Gruenwald" -> "lgruenwald"
        "M. Soheil Hemmati" -> "mhemmati"

    Args:
        name: The faculty member's full name.

    Returns:
        str: The generated faculty_id.
    """
    if not name or not name.strip():
        return "unknown"

    # Clean the name
    cleaned = name.strip()

    # Remove academic titles and prefixes
    prefixes = [
        r'^Dr\.?\s+', r'^Prof\.?\s+', r'^Professor\s+',
        r'^Assoc\.?\s+Prof\.?\s+', r'^Associate\s+Professor\s+',
        r'^Asst\.?\s+Prof\.?\s+', r'^Assistant\s+Professor\s+',
        r'^Mr\.?\s+', r'^Ms\.?\s+', r'^Mrs\.?\s+'
    ]
    for prefix in prefixes:
        cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE)

    # Split into parts
    parts = cleaned.split()

    if len(parts) == 0:
        return "unknown"
    elif len(parts) == 1:
        # Only one name (unusual), use the whole thing
        return parts[0].lower().replace('.', '').replace(',', '')
    else:
        # Get first initial (from first non-initial word)
        first_initial = None
        last_name = None

        # Find first word that's not just an initial (length > 1 after removing punctuation)
        for i, part in enumerate(parts):
            clean_part = part.replace('.', '').replace(',', '')
            if len(clean_part) > 1:
                if first_initial is None:
                    first_initial = clean_part[0].lower()
                # Last name is the last word
                last_name = parts[-1].replace('.', '').replace(',', '').lower()
                break

        if first_initial and last_name:
            faculty_id = f"{first_initial}{last_name}"
        elif last_name:
            faculty_id = last_name
        else:
            faculty_id = parts[0].lower().replace('.', '').replace(',', '')

        # Remove any remaining special characters
        faculty_id = re.sub(r'[^a-z0-9]', '', faculty_id)

        return faculty_id


def ensure_unique_faculty_ids(names: list) -> Dict[str, str]:
    """
    Generate unique faculty IDs for a list of names, handling collisions.

    Args:
        names: List of faculty names.

    Returns:
        Dict[str, str]: Mapping of name -> unique faculty_id
    """
    name_to_id = {}
    id_counts = {}

    for name in names:
        base_id = generate_faculty_id(name)

        # Check for collision
        if base_id in id_counts:
            # Add a number suffix
            id_counts[base_id] += 1
            unique_id = f"{base_id}{id_counts[base_id]}"
        else:
            id_counts[base_id] = 1
            unique_id = base_id

        name_to_id[name] = unique_id

    return name_to_id


def map_name_to_faculty_id(
    faculty_name: str,
    csv_faculty_data: Dict[str, str]
) -> Optional[str]:
    """
    Map a faculty name (e.g., from PDF) to a faculty_id using name matching.

    Args:
        faculty_name: The faculty name to map.
        csv_faculty_data: Dict mapping CSV names to their faculty_ids.

    Returns:
        Optional[str]: The matched faculty_id, or None if no match found.
    """
    if not faculty_name:
        return None

    # Try to find a matching name in the CSV data
    for csv_name, faculty_id in csv_faculty_data.items():
        if names_match(faculty_name, csv_name):
            return faculty_id

    return None


def load_faculty_id_mapping(csv_path: str = None) -> Dict[str, str]:
    """
    Load faculty_id mapping from CSV file.

    Args:
        csv_path: Path to the CSV file. If None, uses Config.CSV_PATH.

    Returns:
        Dict[str, str]: Mapping of faculty name -> faculty_id
    """
    import pandas as pd
    from pathlib import Path
    from src.utils.config import Config

    path = Path(csv_path) if csv_path else Config.CSV_PATH

    if not path.exists():
        return {}

    df = pd.read_csv(path)

    name_to_id = {}

    for _, row in df.iterrows():
        if 'Name' in row and 'faculty_id' in row:
            name = str(row['Name'])
            faculty_id = str(row['faculty_id'])
            if name and faculty_id:
                name_to_id[name] = faculty_id

    return name_to_id

