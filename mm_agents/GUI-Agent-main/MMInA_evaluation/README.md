# Evaluation System

This module provides a simplified evaluation system for the GUI Agent, based on the MMInA evaluation harness but adapted for the new framework.

## Overview

The evaluation system supports multiple evaluation types and uses vLLM with Qwen-2.5-7B for LLM-based fuzzy matching when needed.

## Components

### Evaluator Classes

#### `StringEvaluator`
Evaluates text-based answers using:
- **Exact Match**: Checks if the answer exactly matches the reference
- **Must Include**: Checks if all required phrases are in the answer
- **Fuzzy Match**: Uses vLLM with Qwen-2.5-7B to determine semantic similarity

#### `URLExactEvaluator`
Evaluates URL matching using:
- **EXACT**: Checks if the current URL exactly matches reference URLs
- **GOLD in PRED**: Checks if any reference URL is contained in the current URL

#### `HTMLContentEvaluator`
Evaluates page content using:
- **Program HTML**: Checks if specific content appears on the page
- **JavaScript Locators**: Uses JavaScript to select and check elements
- **Multiple Targets**: Supports checking multiple URLs and content requirements

### Helper Functions

#### `create_vllm_client()`
Creates a vLLM client for LLM-based evaluation:
```python
vllm_client = create_vllm_client("http://localhost:8000")
```

#### `llm_fuzzy_match_vllm()`
Performs fuzzy matching using vLLM:
```python
score = llm_fuzzy_match_vllm(prediction, reference, question, vllm_client)
```

## Usage

### Basic Evaluation
```python
from evaluation.evaluator import evaluator_router
from evaluation.helper_functions import create_vllm_client

# Create vLLM client
vllm_client = create_vllm_client()

# Get evaluator for config file
evaluator = evaluator_router("task_config.json", vllm_client)

# Evaluate trajectory
score = evaluator(trajectory, "task_config.json", page, client)
```

### Configuration File Format
```json
{
    "intent": "Find the latest order",
    "eval": {
        "eval_types": ["string_match", "url_match"],
        "reference_answers": {
            "exact_match": "Order #12345",
            "must_include": ["order", "12345"],
            "fuzzy_match": ["latest order information"]
        },
        "reference_url": "https://example.com/orders/12345",
        "url_note": "EXACT"
    }
}
```

## Supported Evaluation Types

### `string_match`
Evaluates text answers using exact match, must include, and fuzzy match.

### `url_match`
Evaluates URL matching with exact or partial matching rules.

### `program_html`
Evaluates page content using JavaScript selectors and content checks.

## vLLM Integration

The evaluation system uses vLLM with Qwen-2.5-7B for fuzzy matching:

1. **Server Setup**: Ensure vLLM server is running on `http://localhost:8000`
2. **Model**: Uses `qwen2.5-7b-instruct` model
3. **Fallback**: Falls back to simple string matching if vLLM is unavailable

## Error Handling

- **Config Validation**: Validates required fields in config files
- **vLLM Fallback**: Gracefully handles vLLM server unavailability
- **Trajectory Validation**: Checks trajectory structure before evaluation

## Benefits

1. **Simplified**: Removed complex evaluation logic from MMInA
2. **vLLM Integration**: Uses modern LLM for fuzzy matching
3. **Modular**: Easy to add new evaluation types
4. **Robust**: Comprehensive error handling and fallbacks
5. **Compatible**: Works with existing MMInA config files

## Future Extensions

To add new evaluation types:
1. Create a new evaluator class inheriting from `Evaluator`
2. Implement the `__call__` method
3. Add the evaluation type to `evaluator_router`
4. Update config file format documentation 