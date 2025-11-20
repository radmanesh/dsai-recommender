"""Utility functions for matching and normalizing faculty names."""

import re
from typing import Optional


def normalize_faculty_name(name: Optional[str]) -> str:
    """
    Normalize a faculty name for matching.

    Handles variations like:
    - "Dr. John Doe" -> "john doe"
    - "John A. Doe" -> "john doe"
    - "JOHN DOE" -> "john doe"
    - "Doe, John" -> "john doe"

    Args:
        name: The faculty name to normalize.

    Returns:
        str: Normalized name (lowercase, no punctuation, no prefixes).
    """
    if not name:
        return ""

    # Convert to lowercase
    normalized = name.lower()

    # Remove common academic prefixes
    prefixes = [
        r'\bdr\.?\s+',
        r'\bprof\.?\s+',
        r'\bprofessor\s+',
        r'\bassoc\.?\s+prof\.?\s+',
        r'\bassociate\s+professor\s+',
        r'\basst\.?\s+prof\.?\s+',
        r'\bassistant\s+professor\s+',
        r'\bmr\.?\s+',
        r'\bms\.?\s+',
        r'\bmrs\.?\s+',
    ]
    for prefix in prefixes:
        normalized = re.sub(prefix, '', normalized)

    # Handle "Last, First" format -> "First Last"
    if ',' in normalized:
        parts = normalized.split(',')
        if len(parts) == 2:
            # Reverse to "First Last"
            normalized = f"{parts[1].strip()} {parts[0].strip()}"

    # Remove all punctuation (periods, commas, hyphens, etc.)
    # Keep only letters and spaces
    normalized = re.sub(r'[^a-z\s]', ' ', normalized)

    # Normalize whitespace (remove extra spaces, middle initials become single spaces)
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    # Remove single letters (middle initials) that are standalone
    # e.g., "john a doe" -> "john doe"
    words = normalized.split()
    words = [w for w in words if len(w) > 1]
    normalized = ' '.join(words)

    return normalized


def names_match(name1: Optional[str], name2: Optional[str]) -> bool:
    """
    Check if two faculty names match after normalization.

    Args:
        name1: First name to compare.
        name2: Second name to compare.

    Returns:
        bool: True if names match, False otherwise.
    """
    if not name1 or not name2:
        return False

    norm1 = normalize_faculty_name(name1)
    norm2 = normalize_faculty_name(name2)

    if not norm1 or not norm2:
        return False

    # Direct match
    if norm1 == norm2:
        return True

    # Split into words
    words1 = norm1.split()
    words2 = norm2.split()

    # If both have at least 2 words, check last name match and first name similarity
    if len(words1) >= 2 and len(words2) >= 2:
        # Last names must match exactly (after normalization)
        last1 = words1[-1]
        last2 = words2[-1]

        if last1 == last2:
            # Last names match, now check first names
            first1 = words1[0]
            first2 = words2[0]

            # Direct first name match
            if first1 == first2:
                return True

            # Check for common spelling variations (e.g., Dimitrios vs Dimitris)
            # Calculate similarity: if one is a prefix of the other and length difference <= 2
            if len(first1) >= 3 and len(first2) >= 3:
                if first1.startswith(first2[:3]) or first2.startswith(first1[:3]):
                    # Check if they're similar (Levenshtein-like: same first 3 chars)
                    if abs(len(first1) - len(first2)) <= 2:
                        return True

            # If last name matches and first names are similar length, consider it a match
            # This handles cases like "Dimitrios" vs "Dimitris"
            if abs(len(first1) - len(first2)) <= 2 and min(len(first1), len(first2)) >= 4:
                # Check if they share at least 4 characters from the start
                min_len = min(len(first1), len(first2))
                if first1[:min_len] == first2[:min_len]:
                    return True

    # Check if one is a subset of the other (handles cases like "John Doe" vs "John Michael Doe")
    set1 = set(words1)
    set2 = set(words2)

    # If one set of words is a subset of the other, consider it a match
    # (but require at least 2 words to match to avoid false positives)
    if len(set1) >= 2 and len(set2) >= 2:
        if set1.issubset(set2) or set2.issubset(set1):
            return True

    return False


def fuzzy_match_names(name: str, candidate_names: list[str]) -> Optional[str]:
    """
    Find the best matching name from a list of candidates.

    Args:
        name: The name to match.
        candidate_names: List of candidate names to match against.

    Returns:
        Optional[str]: The best matching candidate name, or None if no match found.
    """
    if not name or not candidate_names:
        return None

    normalized_target = normalize_faculty_name(name)

    for candidate in candidate_names:
        if names_match(name, candidate):
            return candidate

    return None

