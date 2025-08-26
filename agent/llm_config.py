"""LLM configuration for different model types"""
import argparse
from typing import Dict, Any


def configure_vllm_llm(args: argparse.Namespace) -> Dict[str, Any]:
    """Configure LLM for vLLM with qwen2.5-instruct-VL"""
    model_name_map = {
        'qwen2.5-vl': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'websailor': 'Alibaba-NLP/WebSailor-7B',
        
    }
    model_server_map = {
        'qwen2.5-vl': 'http://localhost:8000/v1',
        'websailor': 'http://localhost:8002/v1',
    }
    llm_config = {
        'model_type': 'qwenvl_oai',
        'model': model_name_map[args.model],
        'model_server': model_server_map[args.model],  # vLLM server
        'api_key': 'EMPTY',
        'generate_cfg': {
            'max_retries': 10,
            'fncall_prompt_type': 'nous',
            'temperature': 0.2,
            'top_p': 0.9,
            'max_tokens': args.max_tokens,
            'stop': [],
        }
    }
    return llm_config


def configure_huggingface_llm(args: argparse.Namespace) -> Dict[str, Any]:
    """Configure LLM for HuggingFace models (placeholder for future implementation)"""
    # TODO: Implement HuggingFace configuration
    raise NotImplementedError("HuggingFace LLM configuration not yet implemented")


def configure_openai_llm(args: argparse.Namespace) -> Dict[str, Any]:
    """Configure LLM for OpenAI models"""
    llm_config = {
        'model_type': 'openai',
        'model': args.model,  # Use the model name directly from args
        'model_type': 'oai',
        'api_key': args.openai_api_key if hasattr(args, 'openai_api_key') else None,
        'base_url': 'https://api.openai.com/v1',
        'generate_cfg': {
            'max_retries': 10,
            'fncall_prompt_type': 'nous',
            'temperature': 0.2,
            'top_p': 0.9,
            'max_tokens': args.max_tokens,
            'stop': [],
        }
    }
    return llm_config


def load_grounding_model_vllm(args: argparse.Namespace):
    """
    Load grounding model using vLLM server with OpenAI client.
    
    Args:
        args: Arguments object
        
    Returns:
        Grounding model client
    """
    from openai import OpenAI
    
    # Create client with custom base URL pointing to your vLLM server
    grounding_model = OpenAI(
        base_url="http://localhost:8001/v1",  # Adjust the port if needed
        api_key="dummy-key"  # vLLM doesn't check API keys, but the client requires one
    )
    
    return grounding_model


def configure_llm(args: argparse.Namespace) -> Dict[str, Any]:
    """Configure LLM based on the specified model type"""
    if args.model in ['gpt-4o', 'gpt-4o-mini']:
        return configure_openai_llm(args)
    else:
        # Default to vLLM
        return configure_vllm_llm(args) 