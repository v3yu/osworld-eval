"""LLM wrapper to capture actual model inputs including function prompts"""

import copy
from typing import Dict, List, Any, Optional, Iterator, Union
from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.schema import Message
from utils.training_data_collector import get_collector


class LLMWrapper(BaseChatModel):
    """Wrapper around LLM to capture actual model inputs"""
    
    def __init__(self, llm: BaseChatModel):
        """
        Initialize the LLM wrapper
        
        Args:
            llm: The underlying LLM instance
        """
        # Initialize the base class with the LLM's config
        super().__init__(cfg={})
        self.llm = llm
        self.collector = get_collector()
        
        # Copy important attributes from the wrapped LLM
        self.model = getattr(llm, 'model', '')
        self.model_type = getattr(llm, 'model_type', '')
        self.generate_cfg = getattr(llm, 'generate_cfg', {})
        # self.support_multimodal_input = True
        # self.support_multimodal_output = False
        # self.support_audio_input = False
    
    def chat(self, 
             messages: List[Union[Any, Dict]], 
             functions: Optional[List[Dict]] = None,
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
        # Store original messages for comparison
        original_messages = copy.deepcopy(messages)
        
        # Convert messages to list of dicts for easier handling
        original_messages_dict = []
        for msg in messages:
            if Message is not None and isinstance(msg, Message):
                if hasattr(msg, 'model_dump'):
                    original_messages_dict.append(msg.model_dump())
                else:
                    original_messages_dict.append(msg.__dict__)
            elif isinstance(msg, dict):
                original_messages_dict.append(copy.deepcopy(msg))
            else:
                # Fallback: try to convert to dict
                try:
                    original_messages_dict.append(msg.__dict__)
                except:
                    original_messages_dict.append(str(msg))
        
        # Call the original LLM
        response = self.llm.chat(
            messages=messages,
            functions=functions,
            stream=stream,
            delta_stream=delta_stream,
            extra_generate_cfg=extra_generate_cfg,
            **kwargs
        )
        
        # Capture the actual model input if collector is enabled
        if self.collector and self.collector.enabled:
            try:
                # Get the actual model input by accessing the preprocessed messages
                actual_model_input = self._get_actual_model_input(messages, functions, extra_generate_cfg)
                
                # Add to conversation if one is active, otherwise collect as single interaction
                if hasattr(self.collector, 'current_conversation_id') and self.collector.current_conversation_id:
                    self.collector.add_conversation_round(
                        messages=original_messages_dict,
                        response=response,
                        actual_model_input=actual_model_input,
                        functions=functions,
                        round_info={
                            "stream": stream,
                            "delta_stream": delta_stream,
                            "extra_generate_cfg": extra_generate_cfg,
                            **kwargs
                        }
                    )
                else:
                    # Collect the complete interaction with actual model input
                    self.collector.collect_interaction_with_actual_input(
                        original_messages=original_messages_dict,
                        actual_model_input=actual_model_input,
                        response=response,
                        functions=functions,
                        context={
                            "stream": stream,
                            "delta_stream": delta_stream,
                            "extra_generate_cfg": extra_generate_cfg,
                            **kwargs
                        }
                    )
            except Exception as e:
                print(f"Warning: Could not capture actual model input: {e}")
        
        return response
    
    def _get_actual_model_input(self, 
                               messages: List[Union[Any, Dict]], 
                               functions: Optional[List[Dict]] = None,
                               extra_generate_cfg: Optional[Dict] = None) -> List[Dict]:
        """
        Get the actual input sent to the model (after preprocessing)
        
        Args:
            messages: Original messages
            functions: Available functions
            extra_generate_cfg: Extra generation configuration
            
        Returns:
            Actual model input as list of dicts
        """
        try:
            # Convert messages to proper Message objects if needed
            from qwen_agent.llm.schema import Message
            
            # Convert dict messages to Message objects
            message_objects = []
            for msg in messages:
                if isinstance(msg, dict):
                    message_objects.append(Message(**msg))
                elif isinstance(msg, Message):
                    message_objects.append(msg)
                else:
                    # Try to convert other types
                    try:
                        message_objects.append(Message(**msg.__dict__))
                    except:
                        print(f"Warning: Could not convert message {type(msg)} to Message object")
                        continue
            
            if not message_objects:
                print("Warning: No valid messages to preprocess")
                return self._convert_messages_to_dicts(messages)
            
            # Get the language setting
            lang = 'en'
            if extra_generate_cfg and 'lang' in extra_generate_cfg:
                lang = extra_generate_cfg['lang']
            
            # Call the preprocess method to get the actual input
            if hasattr(self.llm, '_preprocess_messages'):
                try:
                    preprocessed_messages = self.llm._preprocess_messages(
                        messages=message_objects,
                        lang=lang,
                        generate_cfg=extra_generate_cfg or {},
                        functions=functions
                    )
                    
                    # Convert preprocessed messages to dict format
                    return self._convert_messages_to_dicts(preprocessed_messages)
                    
                except Exception as e:
                    print(f"Warning: Error in _preprocess_messages: {e}")
                    return self._convert_messages_to_dicts(messages)
            else:
                print("Warning: LLM does not have _preprocess_messages method")
                return self._convert_messages_to_dicts(messages)
                
        except Exception as e:
            print(f"Warning: Could not get actual model input: {e}")
            return self._convert_messages_to_dicts(messages)
    
    def _convert_messages_to_dicts(self, messages: List[Union[Any, Dict]]) -> List[Dict]:
        """Convert messages to dictionary format"""
        actual_input = []
        for msg in messages:
            if isinstance(msg, dict):
                actual_input.append(copy.deepcopy(msg))
            elif hasattr(msg, 'model_dump'):
                actual_input.append(msg.model_dump())
            elif hasattr(msg, '__dict__'):
                actual_input.append(msg.__dict__)
            else:
                actual_input.append(str(msg))
        return actual_input
    
    def _chat(self, messages, stream, delta_stream, generate_cfg):
        """Delegate to the wrapped LLM"""
        return self.llm._chat(messages, stream, delta_stream, generate_cfg)
    
    def _chat_no_stream(self, messages, generate_cfg):
        """Delegate to the wrapped LLM"""
        return self.llm._chat_no_stream(messages, generate_cfg)
    
    def _chat_stream(self, messages, delta_stream, generate_cfg):
        """Delegate to the wrapped LLM"""
        return self.llm._chat_stream(messages, delta_stream, generate_cfg)
    
    def _chat_with_functions(self, messages, functions, stream, delta_stream, generate_cfg, lang):
        """Delegate to the wrapped LLM"""
        return self.llm._chat_with_functions(messages, functions, stream, delta_stream, generate_cfg, lang)
    
    def _continue_assistant_response(self, messages, generate_cfg, stream):
        """Delegate to the wrapped LLM"""
        return self.llm._continue_assistant_response(messages, generate_cfg, stream)
    
    def _preprocess_messages(self, messages, lang, generate_cfg, functions):
        """Delegate to the wrapped LLM"""
        return self.llm._preprocess_messages(messages, lang, generate_cfg, functions)
    
    def _postprocess_messages(self, messages, fncall_mode, generate_cfg):
        """Delegate to the wrapped LLM"""
        return self.llm._postprocess_messages(messages, fncall_mode, generate_cfg)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the underlying LLM"""
        return getattr(self.llm, name)


def wrap_llm(llm: BaseChatModel) -> LLMWrapper:
    """
    Wrap an LLM instance to capture actual model inputs
    
    Args:
        llm: The LLM instance to wrap
        
    Returns:
        Wrapped LLM instance
    """
    return LLMWrapper(llm) 