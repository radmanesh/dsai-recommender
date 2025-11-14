"""CSV loader for faculty metadata."""

import pandas as pd
from pathlib import Path
from typing import List
from llama_index.core import Document
from src.utils.config import Config


def load_faculty_csv(csv_path: str = None) -> List[Document]:
    """
    Load faculty data from CSV and convert to LlamaIndex Documents.

    Args:
        csv_path: Path to the CSV file. Defaults to Config.CSV_PATH.

    Returns:
        List[Document]: List of LlamaIndex Document objects.
    """
    path = Path(csv_path) if csv_path else Config.CSV_PATH

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    print(f"Loading faculty data from: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} faculty records")

    documents = []

    for idx, row in df.iterrows():
        # Create a rich text representation of the faculty member
        text_parts = []

        # Basic info - using actual CSV column names
        if 'Name' in row and pd.notna(row['Name']):
            text_parts.append(f"Faculty Name: {row['Name']}")

        if 'Role' in row and pd.notna(row['Role']):
            text_parts.append(f"Role: {row['Role']}")

        # Research areas (most important for matching)
        if 'Areas' in row and pd.notna(row['Areas']):
            text_parts.append(f"Research Areas: {row['Areas']}")

        # Note: CSV has typo "Research Intresests" instead of "Research Interests"
        if 'Research Intresests' in row and pd.notna(row['Research Intresests']):
            text_parts.append(f"Research Interests: {row['Research Intresests']}")

        # Websites
        if 'Website' in row and pd.notna(row['Website']):
            text_parts.append(f"Website: {row['Website']}")

        if 'Lab Website' in row and pd.notna(row['Lab Website']):
            text_parts.append(f"Lab Website: {row['Lab Website']}")

        # Additional fields (if any)
        for col in df.columns:
            if col not in ['Name', 'Role', 'Areas', 'Research Intresests', 'Website', 'Lab Website']:
                if pd.notna(row[col]):
                    text_parts.append(f"{col}: {row[col]}")

        # Combine all text
        text = "\n".join(text_parts)

        # Create metadata dictionary
        metadata = {
            "source": str(path),
            "csv_row": int(idx),
            "type": "faculty_profile",  # Changed to "faculty_profile" for dual-store architecture
            "faculty_name": str(row.get('Name', 'Unknown')),  # Add for easier filtering
            "faculty_id": str(row.get('faculty_id', ''))  # Add faculty_id for linking
        }

        # Add all available fields to metadata with normalized keys
        for col in df.columns:
            if pd.notna(row[col]):
                # Normalize column names: lowercase and replace spaces with underscores
                key = col.lower().replace(' ', '_')
                metadata[key] = str(row[col])

        # Create Document
        doc = Document(
            text=text,
            metadata=metadata,
        )
        documents.append(doc)

    print(f"Created {len(documents)} documents from CSV")
    return documents


def validate_csv_format(csv_path: str = None) -> dict:
    """
    Validate the CSV file format and return information about it.

    Args:
        csv_path: Path to the CSV file. Defaults to Config.CSV_PATH.

    Returns:
        dict: Information about the CSV file.
    """
    path = Path(csv_path) if csv_path else Config.CSV_PATH

    if not path.exists():
        return {"valid": False, "error": f"File not found: {path}"}

    try:
        df = pd.read_csv(path)

        # Check for recommended columns
        recommended_cols = ['name', 'role', 'department', 'areas', 'research_interests']
        missing_cols = [col for col in recommended_cols if col not in df.columns]

        info = {
            "valid": True,
            "path": str(path),
            "rows": len(df),
            "columns": list(df.columns),
            "recommended_columns": recommended_cols,
            "missing_recommended_columns": missing_cols,
            "sample_row": df.iloc[0].to_dict() if len(df) > 0 else None,
        }

        return info

    except Exception as e:
        return {"valid": False, "error": str(e)}

