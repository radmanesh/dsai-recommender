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

        # Basic info
        if 'name' in row and pd.notna(row['name']):
            text_parts.append(f"Faculty Name: {row['name']}")

        if 'role' in row and pd.notna(row['role']):
            text_parts.append(f"Role: {row['role']}")

        if 'department' in row and pd.notna(row['department']):
            text_parts.append(f"Department: {row['department']}")

        # Research areas (most important for matching)
        if 'areas' in row and pd.notna(row['areas']):
            text_parts.append(f"Research Areas: {row['areas']}")

        if 'research_interests' in row and pd.notna(row['research_interests']):
            text_parts.append(f"Research Interests: {row['research_interests']}")

        # Contact info
        if 'email' in row and pd.notna(row['email']):
            text_parts.append(f"Email: {row['email']}")

        if 'website' in row and pd.notna(row['website']):
            text_parts.append(f"Website: {row['website']}")

        # Additional fields (if any)
        for col in df.columns:
            if col not in ['name', 'role', 'department', 'areas', 'research_interests', 'email', 'website']:
                if pd.notna(row[col]):
                    text_parts.append(f"{col}: {row[col]}")

        # Combine all text
        text = "\n".join(text_parts)

        # Create metadata dictionary
        metadata = {
            "source": "csv",
            "csv_row": int(idx),
            "type": "faculty_profile",
        }

        # Add all available fields to metadata
        for col in df.columns:
            if pd.notna(row[col]):
                metadata[col] = str(row[col])

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

