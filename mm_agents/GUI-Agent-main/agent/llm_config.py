"""LLM configuration for different model types"""
import argparse
from typing import Dict, Any, List, Optional, Union
import base64
import json
from openai import OpenAI
import requests


class DirectVLLMModel:
    """Direct vLLM model wrapper that can be used without qwen_agent"""
    
    def __init__(self, model_name: str, server_url: str, api_key: str = "EMPTY", **kwargs):
        self.model_name = model_name
        self.server_url = server_url
        self.api_key = api_key
        self.client = OpenAI(
            base_url=server_url,
            api_key=api_key
        )
        self.temperature = kwargs.get('temperature', 0.2)
        self.top_p = kwargs.get('top_p', 0.9)
        self.max_tokens = kwargs.get('max_tokens', 2048)
    
    def chat(self, messages: List[Dict], stream: bool = False, functions: List[Dict] = None, function_call: str = "auto", **kwargs):
        """Chat with the model using simplified message format"""
        # Prepare function calling parameters
        call_params = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": kwargs.get('temperature', self.temperature),
            "top_p": kwargs.get('top_p', self.top_p),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
        }
        
        # # Add function calling if provided
        # if functions:
        #     call_params["functions"] = functions
        #     call_params["function_call"] = function_call
        
        # Call the model
        response = self.client.chat.completions.create(**call_params)
        
        if stream:
            return response
        else:
            return response.choices[0].message


class DirectOpenAIModel:
    """Direct OpenAI model wrapper"""
    
    def __init__(self, model_name: str, api_key: str, base_url: str = "https://api.openai.com/v1", **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.temperature = kwargs.get('temperature', 0.2)
        self.top_p = kwargs.get('top_p', 0.9)
        self.max_tokens = kwargs.get('max_tokens', 2048)
    
    def chat(self, messages: List[Dict], stream: bool = False, functions: List[Dict] = None, function_call: str = "auto", **kwargs):
        """Chat with the model using simplified message format"""
        # Prepare function calling parameters
        call_params = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": kwargs.get('temperature', self.temperature),
            "top_p": kwargs.get('top_p', self.top_p),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
        }
        
        # # Add function calling if provided
        # if functions:
        #     call_params["functions"] = functions
        #     call_params["function_call"] = function_call
        
        # Call the model
        response = self.client.chat.completions.create(**call_params)
        
        if stream:
            return response
        else:
            return response.choices[0].message


def create_direct_vllm_model(args: argparse.Namespace, model_name: str = None) -> DirectVLLMModel:
    """Create a direct vLLM model instance"""
    model_name_map = {
        'qwen2.5-vl': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'websailor': 'Alibaba-NLP/WebSailor-7B',
        'ui-tars': 'ByteDance-Seed/UI-TARS-1.5-7B',
    }
    model_server_map = {
        'qwen2.5-vl': 'http://localhost:8000/v1',
        'websailor': 'http://localhost:8002/v1',
        'ui-tars': 'http://localhost:8001/v1',
    }
    if model_name is None:
        model_name = model_name_map.get(args.model, args.model)
        server_url = model_server_map.get(args.model, 'http://localhost:8000/v1')
    else:
        model_name = model_name_map.get(model_name, model_name)
        server_url = model_server_map.get(model_name, 'http://localhost:8000/v1')
    
    return DirectVLLMModel(
        model_name=model_name,
        server_url=server_url,
        api_key="EMPTY",
        temperature=0.2,
        top_p=0.9,
        max_tokens=args.max_tokens,
    )


def create_direct_openai_model(args: argparse.Namespace, model_name: str = None) -> DirectOpenAIModel:
    """Create a direct OpenAI model instance"""
    if model_name is None:
        model_name = args.model
    return DirectOpenAIModel(
        model_name=model_name,
        api_key=args.openai_api_key if hasattr(args, 'openai_api_key') else None,
        base_url="https://api.openai.com/v1",
        temperature=0.2,
        top_p=0.9,
        max_tokens=args.max_tokens,
    )


def create_direct_model(args: argparse.Namespace):
    """Create a direct model instance based on model type"""
    if args.model in ['gpt-4o', 'gpt-4o-mini']:
        return create_direct_openai_model(args)
    else:
        # Default to vLLM
        return create_direct_vllm_model(args)


def configure_vllm_llm(args: argparse.Namespace) -> Dict[str, Any]:
    """Configure LLM for vLLM with qwen2.5-instruct-VL"""
    model_name_map = {
        'qwen2.5-vl': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'websailor': 'Alibaba-NLP/WebSailor-7B',
        'ui-tars': 'ByteDance-Seed/UI-TARS-1.5-7B',
        
    }
    model_server_map = {
        'qwen2.5-vl': 'http://localhost:8000/v1',
        'websailor': 'http://localhost:8002/v1',
        'ui-tars': 'http://localhost:8001/v1',
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
        }
    }
    return llm_config

def configure_claude_llm(args: argparse.Namespace) -> Dict[str, Any]:
    """Configure LLM for Anthropic Claude models"""
    llm_config = {
        'model_type': 'oai',
        'model': args.model,  # Use the model name directly from args
        'api_key': args.anthropic_api_key if hasattr(args, 'anthropic_api_key') else None,
        'base_url': 'https://api.anthropic.com/v1',
        'model_server': 'https://api.anthropic.com/v1',
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

def load_tool_llm(args: argparse.Namespace) -> DirectVLLMModel:
    """Load tool LLM"""
    # tool_model = create_direct_vllm_model(args, model_name='qwen2.5-vl')
    tool_model = create_direct_openai_model(args, model_name='gpt-4o-mini')
    return tool_model

def configure_llm(args: argparse.Namespace) -> Dict[str, Any]:
    """Configure LLM based on the specified model type"""
    if args.model in ['gpt-4o', 'gpt-4o-mini']:
        return configure_openai_llm(args)
    if args.model in ['claude-3-5-sonnet-20240620']:
        return configure_claude_llm(args)
    else:
        # Default to vLLM
        return configure_vllm_llm(args) 