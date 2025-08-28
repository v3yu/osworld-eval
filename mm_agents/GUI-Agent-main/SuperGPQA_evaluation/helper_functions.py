"""Helper functions for SuperGPQA evaluation"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_supergpqa_data(data_path: str = "hf://datasets/m-a-p/SuperGPQA/SuperGPQA-all.jsonl") -> pd.DataFrame:
    """Load SuperGPQA dataset from HuggingFace"""
    try:
        # Load the dataset using pandas
        df = pd.read_json(data_path, lines=True)
        return df
    except Exception as e:
        print(f"Error loading SuperGPQA data: {e}")
        print("Make sure you're logged in with `huggingface-cli login`")
        return pd.DataFrame()


def clean_answer(answer: str) -> str:
    """Clean and normalize answer text"""
    if not answer:
        return ""
    
    # Remove extra whitespace and normalize
    cleaned = answer.strip()
    
    # Remove common prefixes/suffixes
    cleaned = cleaned.replace("Answer:", "").replace("answer:", "").strip()
    
    return cleaned


def extract_question_from_config(configs: Dict[str, Any]) -> str:
    """Extract question from SuperGPQA config"""
    return configs.get("question", "")


def extract_options_from_config(configs: Dict[str, Any]) -> List[str]:
    """Extract options from SuperGPQA config"""
    return configs.get("options", [])


def extract_answer_from_config(configs: Dict[str, Any]) -> str:
    """Extract correct answer from SuperGPQA config"""
    return configs.get("answer", "")


def extract_answer_letter_from_config(configs: Dict[str, Any]) -> str:
    """Extract answer letter from SuperGPQA config"""
    return configs.get("answer_letter", "")


def extract_discipline_from_config(configs: Dict[str, Any]) -> str:
    """Extract discipline from SuperGPQA config"""
    return configs.get("discipline", "")


def extract_field_from_config(configs: Dict[str, Any]) -> str:
    """Extract field from SuperGPQA config"""
    return configs.get("field", "")


def extract_subfield_from_config(configs: Dict[str, Any]) -> str:
    """Extract subfield from SuperGPQA config"""
    return configs.get("subfield", "")


def extract_difficulty_from_config(configs: Dict[str, Any]) -> str:
    """Extract difficulty from SuperGPQA config"""
    return configs.get("difficulty", "")


def extract_is_calculation_from_config(configs: Dict[str, Any]) -> bool:
    """Extract is_calculation flag from SuperGPQA config"""
    return configs.get("is_calculation", False)


def create_supergpqa_config(
    question: str,
    options: List[str],
    answer: str,
    answer_letter: str,
    discipline: str = "",
    field: str = "",
    subfield: str = "",
    difficulty: str = "",
    is_calculation: bool = False,
    task_id: str = "",
    start_url: str = "https://www.bing.com"
) -> Dict[str, Any]:
    """Create a SuperGPQA config dictionary"""
    
    config = {
        "question": question,
        "options": options,
        "answer": answer,
        "answer_letter": answer_letter,
        "discipline": discipline,
        "field": field,
        "subfield": subfield,
        "difficulty": difficulty,
        "is_calculation": is_calculation,
        "task_id": task_id,
        "start_url": start_url,
        "task_type": "supergpqa"
    }
    
    return config


def save_supergpqa_config(config: Dict[str, Any], output_path: Path) -> None:
    """Save SuperGPQA config to JSON file"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_supergpqa_config(config_path: Path) -> Dict[str, Any]:
    """Load SuperGPQA config from JSON file"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_question_with_options(question: str, options: List[str]) -> str:
    """Format question with options for display"""
    formatted = f"Question: {question}\n\nOptions:\n"
    
    for i, option in enumerate(options):
        letter = chr(65 + i)  # A, B, C, D, etc.
        formatted += f"{letter}. {option}\n"
    
    return formatted


def get_option_by_letter(options: List[str], letter: str) -> Optional[str]:
    """Get option text by letter (A, B, C, D, etc.)"""
    if not letter or not options:
        return None
    
    # Convert letter to index (A=0, B=1, C=2, etc.)
    try:
        index = ord(letter.upper()) - ord('A')
        if 0 <= index < len(options):
            return options[index]
    except (ValueError, IndexError):
        pass
    
    return None


def validate_supergpqa_data(row: pd.Series) -> bool:
    """Validate that a SuperGPQA data row has required fields"""
    required_fields = ["question", "options", "answer", "answer_letter"]
    
    for field in required_fields:
        # Check if field exists in the row
        if field not in row.index:
            return False
        
        # Check if the field value is NaN (handle both scalar and array cases)
        field_value = row[field]
        if pd.isna(field_value).any() if hasattr(pd.isna(field_value), 'any') else pd.isna(field_value):
            return False
    
    # Check that options is a list and has at least 2 options
    options = row["options"]
    if not isinstance(options, list) or len(options) < 2:
        return False
    
    # Check that answer_letter corresponds to an option
    answer_letter = row["answer_letter"]
    if not answer_letter or not get_option_by_letter(options, answer_letter):
        return False
    
    return True 