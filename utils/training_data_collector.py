"""Training data collection utility for the GUI Agent"""

import json
import uuid
import copy
import base64
import gzip
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

class ConversationSaver(ABC):
    """Abstract base class for conversation saving strategies"""
    
    @abstractmethod
    def save(self, conversation_data: Dict[str, Any], filepath: Path) -> bool:
        """Save conversation data to file"""
        pass
    
    @abstractmethod
    def get_extension(self) -> str:
        """Get file extension for this format"""
        pass
    
    @abstractmethod
    def estimate_size(self, conversation_data: Dict[str, Any]) -> int:
        """Estimate file size in bytes"""
        pass

class JSONSaver(ConversationSaver):
    """Save conversations as JSON files"""
    
    def save(self, conversation_data: Dict[str, Any], filepath: Path) -> bool:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False, separators=(',', ': '))
            return True
        except Exception as e:
            print(f"Error saving JSON: {e}")
            return False
    
    def get_extension(self) -> str:
        return ".json"
    
    def estimate_size(self, conversation_data: Dict[str, Any]) -> int:
        return sys.getsizeof(json.dumps(conversation_data, ensure_ascii=False, separators=(',', ': ')))

class CompressedJSONSaver(ConversationSaver):
    """Save conversations as compressed JSON files"""
    
    def save(self, conversation_data: Dict[str, Any], filepath: Path) -> bool:
        try:
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False, separators=(',', ': '))
            return True
        except Exception as e:
            print(f"Error saving compressed JSON: {e}")
            return False
    
    def get_extension(self) -> str:
        return ".json.gz"
    
    def estimate_size(self, conversation_data: Dict[str, Any]) -> int:
        # Estimate compressed size (typically 70-90% compression for text, less for base64 images)
        uncompressed_size = sys.getsizeof(json.dumps(conversation_data, ensure_ascii=False, separators=(',', ': ')))
        # Assume 60% compression for mixed content (text + base64 images)
        return int(uncompressed_size * 0.6)

class JSONLSaver(ConversationSaver):
    """Save conversations as JSONL files (one JSON object per line)"""
    
    def save(self, conversation_data: Dict[str, Any], filepath: Path) -> bool:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # Write metadata first
                metadata = {k: v for k, v in conversation_data.items() if k != 'rounds'}
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
                
                # Write each round as a separate line
                for round_data in conversation_data.get('rounds', []):
                    f.write(json.dumps(round_data, ensure_ascii=False) + '\n')
            return True
        except Exception as e:
            print(f"Error saving JSONL: {e}")
            return False
    
    def get_extension(self) -> str:
        return ".jsonl"
    
    def estimate_size(self, conversation_data: Dict[str, Any]) -> int:
        # JSONL is typically slightly larger than JSON due to repeated structure
        return int(sys.getsizeof(json.dumps(conversation_data, ensure_ascii=False, separators=(',', ': '))) * 1.1)

class CompressedJSONLSaver(ConversationSaver):
    """Save conversations as compressed JSONL files"""
    
    def save(self, conversation_data: Dict[str, Any], filepath: Path) -> bool:
        try:
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                # Write metadata first
                metadata = {k: v for k, v in conversation_data.items() if k != 'rounds'}
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
                
                # Write each round as a separate line
                for round_data in conversation_data.get('rounds', []):
                    f.write(json.dumps(round_data, ensure_ascii=False) + '\n')
            return True
        except Exception as e:
            print(f"Error saving compressed JSONL: {e}")
            return False
    
    def get_extension(self) -> str:
        return ".jsonl.gz"
    
    def estimate_size(self, conversation_data: Dict[str, Any]) -> int:
        # Estimate compressed JSONL size
        uncompressed_size = int(sys.getsizeof(json.dumps(conversation_data, ensure_ascii=False, separators=(',', ': '))) * 1.1)
        return int(uncompressed_size * 0.6)

class TrainingDataCollector:
    def __init__(self, output_dir: str = "training_data", enabled: bool = True):
        """
        Initialize the training data collector
        
        Args:
            output_dir: Directory to save training data
            enabled: Whether to enable data collection
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        
        # Session tracking
        self.session_id = str(uuid.uuid4())
        self.session_start = datetime.now()
        
        # Trajectory tracking
        self.current_trajectory_id = None
        self.trajectory_rounds = []
        
        # Conversation tracking
        self.current_conversation_id = None
        self.conversation_history = []
        self.conversation_start_time = None
        self.conversation_task = None
        
        # Use JSONL saver for all conversations
        self.savers = {
            'jsonl': JSONLSaver()
        }
        
        print(f"TrainingDataCollector initialized: output_dir={output_dir}, enabled={enabled}, format=JSONL")
    
    def _select_saver(self, conversation_data: Dict[str, Any]) -> ConversationSaver:
        """
        Always use JSONL format for all conversations
        
        Args:
            conversation_data: The conversation data to save
            
        Returns:
            The JSONL saver
        """
        # Get conversation stats for logging
        total_rounds = len(conversation_data.get('rounds', []))
        has_images = self._has_images(conversation_data)
        
        print(f"Conversation stats: {total_rounds} rounds, has_images={has_images}")
        print("Selected: JSONL (all conversations use JSONL format)")
        
        return self.savers['jsonl']
    
    def _has_images(self, conversation_data: Dict[str, Any]) -> bool:
        """Check if conversation contains images"""
        rounds = conversation_data.get('rounds', [])
        for round_data in rounds:
            messages = round_data.get('messages', [])
            actual_input = round_data.get('actual_model_input', [])
            
            for msg_list in [messages, actual_input]:
                for msg in msg_list:
                    if isinstance(msg, dict) and 'content' in msg:
                        content = msg['content']
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get('type') == 'image_url':
                                    return True
                        elif isinstance(content, str) and 'data:image' in content:
                            return True
        return False
    
    def _optimize_conversation_data(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize conversation data for storage
        
        Args:
            conversation_data: Original conversation data
            
        Returns:
            Optimized conversation data
        """
        optimized_data = copy.deepcopy(conversation_data)
        
        # Optimize each round
        for round_data in optimized_data.get('rounds', []):
            if 'messages' in round_data:
                round_data['messages'] = self._process_messages_with_images(round_data['messages'])
            if 'actual_model_input' in round_data:
                round_data['actual_model_input'] = self._process_messages_with_images(round_data['actual_model_input'])
        
        return optimized_data
    
    def start_conversation(self, conversation_id: str = None, task_description: str = None):
        """
        Start a new conversation collection for a specific task
        
        Args:
            conversation_id: Unique identifier for the conversation
            task_description: Description of the task being performed
        """
        if not self.enabled:
            return
        
        self.current_conversation_id = conversation_id or str(uuid.uuid4())
        self.conversation_history = []
        self.conversation_start_time = datetime.now()
        self.conversation_task = task_description
        
        print(f"Started conversation collection: {self.current_conversation_id}")
        if task_description:
            print(f"Task: {task_description}")
    
    def add_conversation_round(self, 
                             messages: List[Dict[str, Any]], 
                             response: Any, 
                             actual_model_input: Optional[List[Dict[str, Any]]] = None,
                             functions: Optional[List[Dict]] = None,
                             round_info: Optional[Dict[str, Any]] = None):
        """
        Add a round of interaction to the current conversation
        
        Args:
            messages: Input messages for this round
            response: Model response for this round
            actual_model_input: Actual messages sent to the model (after preprocessing)
            functions: Functions that were available
            round_info: Additional information about this round (optional)
        """
        if not self.enabled:
            return
        
        # Process messages to preserve image data
        processed_messages = self._process_messages_with_images(messages)
        
        # Process actual model input if provided
        processed_actual_input = None
        if actual_model_input:
            processed_actual_input = self._process_messages_with_images(actual_model_input)
        
        # Check if this is a page description message that should be skipped
        should_skip = False
        if processed_messages and len(processed_messages) > 0:
            first_msg = processed_messages[0]
            if isinstance(first_msg, dict) and 'content' in first_msg:
                content = first_msg['content']
                if isinstance(content, str) and "You are a helpful assistant that analyzes web page screenshots" in content:
                    should_skip = True
                elif isinstance(content, list) and len(content) > 0 and 'text' in content[0]:
                    if 'You are an expert at parsing natural language responses' in content[0]['text']:
                        should_skip = True
        
        if not should_skip:
            round_data = {
                "round_number": len(self.conversation_history) + 1,
                "timestamp": datetime.now().isoformat(),
                # "messages": processed_messages,
                "actual_model_input": processed_actual_input,
                "response": self._serialize_response(response),
                "functions": functions or [],
                "round_info": round_info or {}
            }
            
            self.conversation_history.append(round_data)
            print(f"Added round {round_data['round_number']} to conversation {self.current_conversation_id}")
    
    def end_conversation(self, conversation_summary: Optional[Dict[str, Any]] = None) -> str:
        """
        End the current conversation and save it to a file using the best format
        
        Args:
            conversation_summary: Additional summary information about the conversation
            
        Returns:
            Path to the saved file
        """
        if not self.enabled or self.current_conversation_id is None:
            return ""
        
        # Prepare the conversation data
        conversation_data = {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "conversation_id": self.current_conversation_id,
            "conversation_start": self.conversation_start_time.isoformat() if self.conversation_start_time else None,
            "conversation_end": datetime.now().isoformat(),
            "task_description": getattr(self, 'conversation_task', None),
            "total_rounds": len(self.conversation_history),
            "rounds": self.conversation_history,
            "conversation_summary": conversation_summary or {},
            "metadata": {
                "data_type": "conversation_history",
                "format_version": "2.0"
            }
        }
        
        # Optimize the data for storage
        optimized_data = self._optimize_conversation_data(conversation_data)
        
        # Select the best saver based on conversation size and content
        saver = self._select_saver(optimized_data)
        
        # Generate filename with appropriate extension
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"conversation_{self.current_conversation_id}{saver.get_extension()}"
        filepath = self.output_dir / filename
        
        # Update metadata
        optimized_data["metadata"]["filename"] = filename
        optimized_data["metadata"]["filepath"] = str(filepath)
        optimized_data["metadata"]["format"] = saver.get_extension()
        
        # Save using the selected saver
        try:
            success = saver.save(optimized_data, filepath)
            if success:
                # Log file size
                actual_size = filepath.stat().st_size if filepath.exists() else 0
                size_mb = actual_size / (1024 * 1024)
                print(f"Conversation saved: {filepath} ({size_mb:.2f} MB)")
                
                # Reset conversation tracking
                self.current_conversation_id = None
                self.conversation_history = []
                self.conversation_start_time = None
                self.conversation_task = None
                
                return str(filepath)
            else:
                print(f"Failed to save conversation using {saver.get_extension()} format")
                return ""
                
        except Exception as e:
            print(f"Error saving conversation: {e}")
            traceback.print_exc()
            return ""
    
    def _process_messages_with_images(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process messages to preserve image data in base64 format and convert ContentItem objects
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Processed messages with image data preserved and ContentItem objects converted
        """
        processed_messages = []
        
        for msg in messages:
            processed_msg = copy.deepcopy(msg)
            
            # Handle different message formats
            if isinstance(processed_msg, dict):
                # Check for image content in the message
                if 'content' in processed_msg:
                    content = processed_msg['content']
                    
                    # Handle list content (multimodal)
                    if isinstance(content, list):
                        processed_content = []
                        for item in content:
                            if isinstance(item, dict):
                                if item.get('type') == 'image_url':
                                    # Preserve image data
                                    image_url = item.get('image_url', {})
                                    if isinstance(image_url, dict) and 'url' in image_url:
                                        # Convert data URL to base64 if needed
                                        url = image_url['url']
                                        if url.startswith('data:image'):
                                            # Already in data URL format, keep as is
                                            processed_content.append(item)
                                        else:
                                            # Try to read file and convert to base64
                                            try:
                                                image_path = Path(url)
                                                if image_path.exists():
                                                    with open(image_path, 'rb') as img_file:
                                                        img_data = img_file.read()
                                                        base64_data = base64.b64encode(img_data).decode('utf-8')
                                                        # Determine mime type from file extension
                                                        mime_type = self._get_mime_type(image_path.suffix)
                                                        data_url = f"data:{mime_type};base64,{base64_data}"
                                                        processed_content.append({
                                                            'type': 'image_url',
                                                            'image_url': {'url': data_url}
                                                        })
                                                    print(f"Converted image to base64: {image_path}")
                                                else:
                                                    # Keep original if file doesn't exist
                                                    processed_content.append(item)
                                            except Exception as e:
                                                print(f"Error processing image {url}: {e}")
                                                processed_content.append(item)
                                    else:
                                        processed_content.append(item)
                                else:
                                    processed_content.append(item)
                            else:
                                # Handle ContentItem objects
                                processed_item = self._convert_content_item(item)
                                processed_content.append(processed_item)
                        processed_msg['content'] = processed_content
                    
                    # Handle string content (check if it contains image references)
                    elif isinstance(content, str):
                        # Keep string content as is for now
                        # Could add logic here to detect and process image references
                        processed_msg['content'] = content
            
            processed_messages.append(processed_msg)
        
        return processed_messages
    
    def _convert_content_item(self, item: Any) -> Dict[str, Any]:
        """
        Convert ContentItem objects to JSON-serializable format
        
        Args:
            item: ContentItem object or other item
            
        Returns:
            JSON-serializable dictionary
        """
        try:
            # Check if it's a ContentItem object with image attribute
            if hasattr(item, 'image') and item.image is not None:
                # It's a ContentItem with image: ContentItem(image=f'data:image/png;base64,{image_base64}')
                return {
                    'type': 'image_url',
                    'image_url': {
                        'url': item.image
                    }
                }
            elif hasattr(item, 'text') and item.text is not None:
                # It's a ContentItem with text: ContentItem(text="some text")
                return {
                    'type': 'text',
                    'text': item.text
                }
            elif hasattr(item, '__dict__'):
                # Try to convert object to dict
                return item.__dict__
            else:
                # Fallback: convert to string
                return str(item)
        except Exception as e:
            print(f"Warning: Could not convert ContentItem: {e}")
            return str(item)
    
    
    
    def _get_mime_type(self, file_extension: str) -> str:
        """Get MIME type from file extension"""
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }
        return mime_types.get(file_extension.lower(), 'image/png')
    
    def start_trajectory(self, trajectory_id: str = None):
        """Start a new trajectory collection"""
        if not self.enabled:
            return
        
        self.current_trajectory_id = trajectory_id or str(uuid.uuid4())
        self.trajectory_rounds = []
        print(f"Started trajectory collection: {self.current_trajectory_id}")
    
    def add_round(self, messages: List[Dict[str, Any]], response: Any, round_info: Optional[Dict[str, Any]] = None):
        """
        Add a round of interaction to the current trajectory
        
        Args:
            messages: Input messages for this round
            response: Model response for this round
            round_info: Additional information about this round (optional)
        """
        if not self.enabled or self.current_trajectory_id is None:
            return
        
        # Process messages to preserve image data
        processed_messages = self._process_messages_with_images(messages)
        
        round_data = {
            "round_number": len(self.trajectory_rounds) + 1,
            "timestamp": datetime.now().isoformat(),
            "messages": processed_messages,
            "response": self._serialize_response(response),
            "round_info": round_info or {}
        }
        
        self.trajectory_rounds.append(round_data)
        print(f"Added round {round_data['round_number']} to trajectory {self.current_trajectory_id}")
    
    def collect_actual_model_input(self, 
                                 original_messages: List[Dict[str, Any]], 
                                 actual_model_input: List[Dict[str, Any]], 
                                 functions: Optional[List[Dict]] = None,
                                 context: Optional[Dict[str, Any]] = None) -> str:
        """
        Collect the actual input sent to the model (including function prompts)
        
        Args:
            original_messages: Original messages before preprocessing
            actual_model_input: Actual messages sent to the model (after preprocessing)
            functions: Functions that were available
            context: Additional context
            
        Returns:
            Path to the saved JSON file
        """
        if not self.enabled:
            return ""
        
        # Process messages to preserve image data
        processed_original = self._process_messages_with_images(original_messages)
        processed_actual = self._process_messages_with_images(actual_model_input)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"model_input_{timestamp}_{uuid.uuid4().hex[:8]}.json"
        filepath = self.output_dir / filename
        
        # Prepare the data structure
        data = {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "timestamp": datetime.now().isoformat(),
            "original_messages": processed_original,
            "actual_model_input": processed_actual,
            "functions": functions or [],
            "context": context or {},
            "metadata": {
                "filename": filename,
                "filepath": str(filepath),
                "input_type": "actual_model_input"
            }
        }
        
        # Save to JSON file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Actual model input saved: {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"Error saving actual model input: {e}")
            return ""
    
    def collect_interaction_with_actual_input(self,
                                            original_messages: List[Dict[str, Any]],
                                            actual_model_input: List[Dict[str, Any]],
                                            response: Any,
                                            functions: Optional[List[Dict]] = None,
                                            context: Optional[Dict[str, Any]] = None) -> str:
        """
        Collect both the actual model input and the response
        
        Args:
            original_messages: Original messages before preprocessing
            actual_model_input: Actual messages sent to the model (after preprocessing)
            response: Model response
            functions: Functions that were available
            context: Additional context
            
        Returns:
            Path to the saved JSON file
        """
        if not self.enabled:
            return ""
        
        # Process messages to preserve image data
        processed_original = self._process_messages_with_images(original_messages)
        processed_actual = self._process_messages_with_images(actual_model_input)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"complete_interaction_{timestamp}_{uuid.uuid4().hex[:8]}.json"
        filepath = self.output_dir / filename
        
        # Prepare the data structure
        data = {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "timestamp": datetime.now().isoformat(),
            "original_messages": processed_original,
            "actual_model_input": processed_actual,
            "response": self._serialize_response(response),
            "functions": functions or [],
            "context": context or {},
            "metadata": {
                "filename": filename,
                "filepath": str(filepath),
                "input_type": "complete_interaction"
            }
        }
        
        # Save to JSON file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Complete interaction saved: {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"Error saving complete interaction: {e}")
            return ""
    
    def end_trajectory(self, trajectory_summary: Optional[Dict[str, Any]] = None) -> str:
        """
        End the current trajectory and save it to a JSON file
        
        Args:
            trajectory_summary: Additional summary information about the trajectory
            
        Returns:
            Path to the saved JSON file
        """
        if not self.enabled or self.current_trajectory_id is None:
            return ""
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"trajectory_{timestamp}_{self.current_trajectory_id[:8]}.json"
        filepath = self.output_dir / filename
        
        # Prepare the trajectory data
        trajectory_data = {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "trajectory_id": self.current_trajectory_id,
            "trajectory_start": self.trajectory_rounds[0]["timestamp"] if self.trajectory_rounds else None,
            "trajectory_end": datetime.now().isoformat(),
            "total_rounds": len(self.trajectory_rounds),
            "rounds": self.trajectory_rounds,
            "trajectory_summary": trajectory_summary or {},
            "metadata": {
                "filename": filename,
                "filepath": str(filepath)
            }
        }
        
        # Save to JSON file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
            print(f"Trajectory saved: {filepath}")
            
            # Reset trajectory tracking
            self.current_trajectory_id = None
            self.trajectory_rounds = []
            
            return str(filepath)
        except Exception as e:
            print(f"Error saving trajectory: {e}")
            return ""
    
    def collect_interaction(
        self,
        messages: List[Dict[str, Any]],
        response: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Collect a single interaction (prompt + response) - for backward compatibility
        
        Args:
            messages: Input messages/prompts
            response: Model response
            context: Additional context (optional)
            
        Returns:
            Path to the saved JSON file
        """
        if not self.enabled:
            return ""
        
        # If we're in a conversation, add this as a round
        if self.current_conversation_id is not None:
            self.add_conversation_round(messages, response, round_info=context)
            return ""
        
        # If we're in a trajectory, add this as a round
        if self.current_trajectory_id is not None:
            self.add_round(messages, response, context)
            return ""
        
        # Otherwise, save as a single interaction (legacy behavior)
        # Process messages to preserve image data
        processed_messages = self._process_messages_with_images(messages)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"interaction_{timestamp}_{uuid.uuid4().hex[:8]}.json"
        filepath = self.output_dir / filename
        
        # Prepare the data structure
        data = {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "timestamp": datetime.now().isoformat(),
            "messages": processed_messages,
            "response": self._serialize_response(response),
            "context": context or {},
            "metadata": {
                "filename": filename,
                "filepath": str(filepath)
            }
        }
        
        # Save to JSON file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Training data saved: {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"Error saving training data: {e}")
            return ""
    
    def _serialize_response(self, response: Any) -> Any:
        """Serialize the response to ensure it's JSON-compatible"""
        if hasattr(response, '__dict__'):
            # Convert objects to dict if possible
            return response.__dict__
        elif isinstance(response, (list, tuple)):
            # Handle list/tuple responses
            return [self._serialize_response(item) for item in response]
        elif hasattr(response, 'content'):
            # Handle response objects with content attribute
            return {
                'content': response.content,
                'type': type(response).__name__
            }
        else:
            # Try to convert to string
            try:
                return str(response)
            except:
                return f"<{type(response).__name__} object>"
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session"""
        if not self.enabled:
            return {}
        
        # Count files by type
        interaction_files = list(self.output_dir.glob("interaction_*.json"))
        trajectory_files = list(self.output_dir.glob("trajectory_*.json"))
        conversation_files = list(self.output_dir.glob("conversation_*.json"))
        model_input_files = list(self.output_dir.glob("model_input_*.json"))
        complete_interaction_files = list(self.output_dir.glob("complete_interaction_*.json"))
        
        total_interactions = len(interaction_files)
        total_trajectories = len(trajectory_files)
        total_conversations = len(conversation_files)
        total_model_inputs = len(model_input_files)
        total_complete_interactions = len(complete_interaction_files)
        
        # Count total rounds across all trajectories
        total_trajectory_rounds = 0
        for filepath in trajectory_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    total_trajectory_rounds += data.get('total_rounds', 0)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        
        # Count total rounds across all conversations
        total_conversation_rounds = 0
        for filepath in conversation_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    total_conversation_rounds += data.get('total_rounds', 0)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        
        return {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "session_duration": str(datetime.now() - self.session_start),
            "total_interactions": total_interactions,
            "total_trajectories": total_trajectories,
            "total_conversations": total_conversations,
            "total_model_inputs": total_model_inputs,
            "total_complete_interactions": total_complete_interactions,
            "total_trajectory_rounds": total_trajectory_rounds,
            "total_conversation_rounds": total_conversation_rounds,
            "output_directory": str(self.output_dir)
        }
    
    def disable(self):
        """Disable data collection"""
        self.enabled = False
        print("Training data collection disabled")
    
    def enable(self):
        """Enable data collection"""
        self.enabled = True
        print(f"Training data collection enabled. Output directory: {self.output_dir}")


# Global instance for easy access
_global_collector = None

def get_collector() -> TrainingDataCollector:
    """Get the global training data collector instance"""
    global _global_collector
    if _global_collector is None:
        _global_collector = TrainingDataCollector()
    return _global_collector

def set_collector(collector: TrainingDataCollector):
    """Set the global training data collector instance"""
    global _global_collector
    _global_collector = collector 