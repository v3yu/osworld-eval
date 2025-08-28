"""LLM wrapper to capture actual model inputs including function prompts"""

import copy
from typing import Dict, List, Any, Optional, Iterator, Union
from utils.training_data_collector import get_collector


class LLMWrapper:
    """Wrapper around LLM to capture actual model inputs"""
    
    def __init__(self, llm):
        """
        Initialize the LLM wrapper
        
        Args:
            llm: The underlying LLM instance
        """
        # Initialize the base class with the LLM's config
        self.llm = llm
        self.collector = get_collector()
        
        # Copy important attributes from the wrapped LLM
        self.model = getattr(llm, 'model', '')
        self.model_type = getattr(llm, 'model_type', '')
        self.generate_cfg = getattr(llm, 'generate_cfg', {})
    
    def chat(self, 
             messages: List[Union[Any, Dict]], 
             stream: bool = True,
             delta_stream: bool = False,
             extra_generate_cfg: Optional[Dict] = None,
             **kwargs) -> Union[List[Any], List[Dict], Iterator[List[Any]], Iterator[List[Dict]]]:
        """
        Chat with the LLM and capture the actual input sent to the model
        
        Args:
            messages: Input messages
            functions: Available functions
            stream: Whether to stream the response
            delta_stream: Whether to use delta streaming
            extra_generate_cfg: Extra generation configuration
            **kwargs: Additional arguments
            
        Returns:
            LLM response
        """
        # Call the original LLM
        response = self.llm.chat(messages=messages)
        
        # Capture the actual model input if collector is enabled
        if self.collector and self.collector.enabled:
            try:
                # Add to conversation if one is active, otherwise collect as single interaction
                if hasattr(self.collector, 'current_conversation_id') and self.collector.current_conversation_id:
                    self.collector.add_conversation_round(
                        messages=messages,
                        response=response,
                        round_info={
                            "stream": stream,
                            "delta_stream": delta_stream,
                            "extra_generate_cfg": extra_generate_cfg,
                            **kwargs
                        }
                    )
            except Exception as e:
                print(f"Warning: Could not capture actual model input: {e}")
        
        return response
    
    # def _chat(self, messages, stream, delta_stream, generate_cfg):
    #     """Delegate to the wrapped LLM"""
    #     return self.llm._chat(messages, stream, delta_stream, generate_cfg)
    
    # def _chat_no_stream(self, messages, generate_cfg):
    #     """Delegate to the wrapped LLM"""
    #     return self.llm._chat_no_stream(messages, generate_cfg)
    
    # def _chat_stream(self, messages, delta_stream, generate_cfg):
    #     """Delegate to the wrapped LLM"""
    #     return self.llm._chat_stream(messages, delta_stream, generate_cfg)
    
    # def _chat_with_functions(self, messages, functions, stream, delta_stream, generate_cfg, lang):
    #     """Delegate to the wrapped LLM"""
    #     return self.llm._chat_with_functions(messages, functions, stream, delta_stream, generate_cfg, lang)
    
    # def _continue_assistant_response(self, messages, generate_cfg, stream):
    #     """Delegate to the wrapped LLM"""
    #     return self.llm._continue_assistant_response(messages, generate_cfg, stream)
    
    # def _preprocess_messages(self, messages, lang, generate_cfg, functions):
    #     """Delegate to the wrapped LLM"""
    #     return self.llm._preprocess_messages(messages, lang, generate_cfg, functions)
    
    # def _postprocess_messages(self, messages, fncall_mode, generate_cfg):
    #     """Delegate to the wrapped LLM"""
    #     return self.llm._postprocess_messages(messages, fncall_mode, generate_cfg)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the underlying LLM"""
        return getattr(self.llm, name)


def wrap_llm(llm) -> LLMWrapper:
    """
    Wrap an LLM instance to capture actual model inputs
    
    Args:
        llm: The LLM instance to wrap
        
    Returns:
        Wrapped LLM instance
    """
    return LLMWrapper(llm) 