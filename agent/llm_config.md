# LLM Configuration Module

This module handles LLM configuration for different model types in the GUI Agent.

## Overview

The `llm_config.py` file provides centralized configuration for different LLM backends, making it easy to switch between vLLM, HuggingFace, and other model types.

## Functions

### `configure_llm(args: argparse.Namespace) -> Dict[str, Any]`
Main configuration function that routes to the appropriate configuration based on the model type.

**Parameters:**
- `args`: Namespace containing model configuration parameters

**Returns:**
- Dictionary with LLM configuration

### `configure_vllm_llm(args: argparse.Namespace) -> Dict[str, Any]`
Configures LLM for vLLM server with qwen2.5-instruct-VL model.

**Configuration includes:**
- Model type: `qwenvl_oai`
- Model: `qwen2.5-instruct-VL`
- Server: `http://localhost:8000/v1`
- Function calling prompt type: `qwen`
- Generation parameters from args (temperature, top_p, max_tokens, etc.)

### `configure_huggingface_llm(args: argparse.Namespace) -> Dict[str, Any]`
Placeholder for HuggingFace model configuration (to be implemented).

## Usage

The agent automatically uses this configuration through the `_configure_llm` method:

```python
from agent import construct_agent

# Create agent with default vLLM configuration
agent = construct_agent(args)

# For HuggingFace models (when implemented)
args.model_type = 'huggingface'
agent = construct_agent(args)
```

## Adding New Model Types

To add support for a new model type:

1. Create a new configuration function (e.g., `configure_openai_llm`)
2. Update the `configure_llm` function to handle the new model type
3. Add the new function to the `__all__` list in `__init__.py`

## Example

```python
def configure_openai_llm(args: argparse.Namespace) -> Dict[str, Any]:
    """Configure LLM for OpenAI models"""
    return {
        'model_type': 'openai',
        'model': 'gpt-4-vision-preview',
        'api_key': args.openai_api_key,
        'generate_cfg': {
            'temperature': args.temperature,
            'max_tokens': args.max_tokens
        }
    }
``` 