"""
GPT-4o model class for convenient usage in scripts
"""
import os
import base64
from typing import List, Dict, Any, Optional, Union
import argparse
from openai import OpenAI


class GPT4o:
    """
    A convenient wrapper class for GPT-4o model interactions
    
    This class provides easy-to-use methods for text and multimodal interactions
    with GPT-4o using the OpenAI API.
    """
    
    def __init__(self, args: argparse.Namespace):
        """
        Initialize GPT-4o client
        
        Args:
            args: Arguments object containing openai_api_key
        """
        self.api_key = args.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set args.openai_api_key")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"
        
    def chat(self, 
             messages: List[Dict[str, Any]], 
             temperature: float = 0.7,
             max_tokens: int = 1000,
             stream: bool = False) -> Union[str, Any]:
        """
        Send a chat completion request to GPT-4o
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated response content or stream object
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                return response
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            print(f"Error in GPT-4o chat: {e}")
            return ""
    
    def chat_with_images(self, 
                        text: str, 
                        image_paths: List[str] = None,
                        image_base64: List[str] = None,
                        temperature: float = 0.7,
                        max_tokens: int = 1000) -> str:
        """
        Send a multimodal chat request with images to GPT-4o
        
        Args:
            text: Text prompt
            image_paths: List of image file paths
            image_base64: List of base64 encoded image strings
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response content
        """
        content = [{"type": "text", "text": text}]
        
        # Add images from file paths
        if image_paths:
            for img_path in image_paths:
                if os.path.exists(img_path):
                    base64_image = self._encode_image_file(img_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
        
        # Add images from base64 strings
        if image_base64:
            for img_b64 in image_base64:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }
                })
        
        messages = [{"role": "user", "content": content}]
        
        return self.chat(messages, temperature, max_tokens)
    
    def _encode_image_file(self, image_path: str) -> str:
        """
        Encode image file to base64 string
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    
    def simple_chat(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Simple text-only chat with GPT-4o
        
        Args:
            prompt: Text prompt
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature)
    
def create_gpt4o_model(args: argparse.Namespace) -> GPT4o:
    """
    Factory function to create a GPT4o model instance
    
    Args:
        args: Arguments object containing openai_api_key
        
    Returns:
        GPT4o model instance
    """
    return GPT4o(args)
