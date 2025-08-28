"""Helper functions for WebWalkerQA evaluation"""

import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Union


def clean_answer(answer: str) -> str:
    """Clean the answer by removing extra whitespace and normalizing"""
    if not isinstance(answer, str):
        return ""
    
    # Remove extra whitespace and normalize
    cleaned = re.sub(r'\s+', ' ', answer.strip())
    return cleaned.lower()


def clean_url(url: str) -> str:
    """Clean the URL by removing trailing slashes and normalizing"""
    if not isinstance(url, str):
        return ""
    
    # Remove trailing slash and normalize
    cleaned = url.rstrip('/')
    return cleaned.lower()


def extract_question_from_config(config: Dict[str, Any]) -> str:
    """Extract question from WebWalkerQA config"""
    return config.get("question", "")


def extract_answer_from_config(config: Dict[str, Any]) -> str:
    """Extract reference answer from WebWalkerQA config"""
    return config.get("answer", "")


def extract_root_url_from_config(config: Dict[str, Any]) -> str:
    """Extract root URL from WebWalkerQA config"""
    return config.get("root_url", "")


def extract_info_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract info metadata from WebWalkerQA config"""
    return config.get("info", {})


def create_webwalkerqa_config(question: str, answer: str, start_url: str, info: Dict[str, Any] = None, task_id: str = "") -> Dict[str, Any]:
    """Create a WebWalkerQA config dictionary"""
    config = {
        "question": question,
        "answer": answer,
        "start_url": start_url,
        "info": info or {},
        "task_id": task_id,
        "task_type": "webwalkerqa"
    }
    return config


def save_webwalkerqa_config(config: Dict[str, Any], output_path: Path) -> None:
    """Save WebWalkerQA config to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    # Convert the config to be JSON serializable
    serializable_config = convert_numpy(config)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_config, f, ensure_ascii=False, indent=2)


def load_webwalkerqa_config(config_path: Path) -> Dict[str, Any]:
    """Load WebWalkerQA config from JSON file"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_webwalkerqa_data(data_path: str = None, split: str = "silver") -> pd.DataFrame:
    """Load WebWalkerQA dataset from HuggingFace"""
    try:
        from datasets import load_dataset
        dataset = load_dataset("callanwu/WebWalkerQA", split=split)
        df = dataset.to_pandas()
        
        print(f"Loaded {len(df)} samples from WebWalkerQA {split} split")
        return df
    except Exception as e:
        print(f"Error loading WebWalkerQA data: {e}")
        # Return empty DataFrame
        return pd.DataFrame()
        

def validate_webwalkerqa_data(row: pd.Series) -> bool:
    """Validate that a WebWalkerQA data row has required fields"""
    required_fields = ["question", "answer", "root_url"]
    
    for field in required_fields:
        # Check if field exists in the row
        if field not in row.index:
            return False
        
        # Check if the field value is NaN (handle both scalar and array cases)
        field_value = row[field]
        if pd.isna(field_value).any() if hasattr(pd.isna(field_value), 'any') else pd.isna(field_value):
            return False
    
    # Check that question and answer are not empty
    if not row["question"] or not row["answer"] or not row["root_url"]:
        return False
    
    return True 