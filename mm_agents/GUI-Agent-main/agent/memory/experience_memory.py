import os
import json
import numpy as np
import faiss
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.help_functions import CLIPTextSimilarity, CLIPMultimodalSimilarity
from actions.help_functions import parse_action_json
from PIL import Image
import base64
from io import BytesIO
from datetime import datetime

LOG_FOLDER = "/home/wenyi/GUI-Agent/log_files"
clip_similarity = CLIPTextSimilarity()

def get_all_subfolders(folder_path):
    all_files = os.listdir(folder_path)
    all_dataset_folders = [item for item in all_files if os.path.isdir(os.path.join(folder_path, item))]
    return all_dataset_folders

def get_unique_values_from_dict(data_dict):
    seen = set()
    unique_values = {}
    
    for key, value_list in data_dict.items():
        # Filter out values already seen in other lists
        unique_for_this_key = [val for val in value_list if val not in seen]
        
        # Add to the result dictionary only if there are unique values
        if unique_for_this_key:
            unique_values[key] = unique_for_this_key
            
        # Update the seen set with all values from this list
        seen.update(value_list)
    
    return unique_values


def get_success_history(log_folder):
    all_dataset_folders = get_all_subfolders(log_folder)
    success_history = {}

    for dataset_folder in all_dataset_folders:
        # print(dataset_folder)
        all_domain_folders = get_all_subfolders(os.path.join(log_folder, dataset_folder))
        for domain_folder in all_domain_folders:
            # print(domain_folder)
            all_log_files = os.listdir(os.path.join(log_folder, dataset_folder, domain_folder))
            all_log_files = [item for item in all_log_files if item.endswith('.log')]
            for log_file in all_log_files:
                if 'test' not in log_file:
                    continue
                model_name = log_file.split('_')[1]
                test_id = log_file.split('_')[-1].split('.')[0]
                success_ids = []
                with open(f"/home/wenyi/GUI-Agent/log_files/{dataset_folder}/{domain_folder}/{log_file}", 'r') as f:
                    log_content = f.read()
                    all_lines = log_content.split('\n')
                    for line in all_lines:
                        if 'PASS' in line:
                            success_ids.append(line.split('/')[-1].split('.')[0])
                if dataset_folder not in success_history:
                    success_history[dataset_folder] = {}
                if domain_folder not in success_history[dataset_folder]:
                    success_history[dataset_folder][domain_folder] = {}
                if test_id not in success_history[dataset_folder][domain_folder]:
                    success_history[dataset_folder][domain_folder][test_id] = list(set(success_ids))
                success_history[dataset_folder][domain_folder] = get_unique_values_from_dict(success_history[dataset_folder][domain_folder])
                
    # print(success_history)
    return success_history
    
    
def get_selected_conversations(current_question, evaluation_type, domain, similar_num):
    if evaluation_type != 'mind2web':
        success_history = get_success_history(LOG_FOLDER)
        success_cases = success_history[evaluation_type][domain]
    else:
        success_cases = {'test_website': [item for item in os.listdir(f'/data/wenyi/training_data/mind2web/{domain}/qwen2.5-vl') if '.jsonl' in item]}
        # print(success_cases)
    success_questions = []
    selected_conversations = []

    for test_id, success_ids in success_cases.items():
        for success_id in success_ids:
            success_id = success_id.replace(' ', '_')
            if evaluation_type != 'mind2web':
                conversation_file = f"/data/wenyi/training_data/{evaluation_type}/{domain}/qwen2.5-vl/{test_id}/conversation_{domain}_{success_id}.jsonl"
            else:
                conversation_file = f"/data/wenyi/training_data/mind2web/{domain}/qwen2.5-vl/{success_id}"
            try:
                with open(conversation_file, 'r') as f:
                    success_questions.append(json.loads(f.readline())['task_description'])
            except:
                print(f"Conversation file {conversation_file} not found")
                continue

        top_similar = clip_similarity.get_top_similar_questions(current_question, success_questions, similar_num)

        for idx, score in top_similar:
            if evaluation_type != 'mind2web':
                conversation_file = f"/data/wenyi/training_data/{evaluation_type}/{domain}/qwen2.5-vl/{test_id}/conversation_{domain}_{success_ids[idx]}.jsonl".replace(' ', '_')
            else:
                conversation_file = f"/data/wenyi/training_data/mind2web/{domain}/qwen2.5-vl/{success_ids[idx]}"
            selected_conversations.append(conversation_file)
            # print(f"Score: {score:.4f} - {success_questions[idx]}")
        if len(top_similar) < similar_num:
            similar_num = similar_num - len(top_similar)
        else:
            break
    # print(selected_conversations)
            
    return selected_conversations


def construct_experience_memory(args, current_question, agent):
    evaluation_type = args.evaluation_type
    domain = args.domain
    similar_num = 3
    selected_conversations = get_selected_conversations(current_question, evaluation_type, domain, similar_num+5)
    action_texts_list = []
    for conversation_file in selected_conversations:
        with open(conversation_file, 'r') as f:
            config_file = json.loads(f.readline())
            task_description = config_file.get('task_description', '')
            # print(task_description)
            if task_description == '':
                print(f"Task description is empty for {conversation_file}")
                continue
            conversation_list = [json.loads(line) for line in f][1:]
        responses_list = [conversation['response'] for conversation in conversation_list]
        actual_actions = []
        previous_action_name, previous_action_reasoning = None, None
        # previous_action_str = ""
        for response in responses_list:
            if isinstance(response, list):
                response = response[0]
            if 'content' in response:
                response = response['content']
            try:
                action_json = parse_action_json(response).get('function_call', {})
                # print(action_json)
                if 'name' in action_json:
                    current_action_name = action_json['name']
                    current_action_reasoning = action_json['arguments']['reasoning']
                elif 'action' in action_json:
                    current_action_name = action_json['action']
                    current_action_reasoning = action_json['reasoning']
                elif 'action_type' in action_json:
                    current_action_name = action_json['action_type']
                    current_action_reasoning = action_json['reasoning']
                elif isinstance(list(action_json.values())[0], dict):
                    current_action_name = list(action_json.keys())[0]
                    current_action_reasoning = list(action_json.values())[0]['reasoning']
                else:
                    print(f"Error: {action_json} has no name, action, or action_type")
                    continue
                action_json['name'] = current_action_name
                action_json['arguments'] = {'reasoning': current_action_reasoning}
                
                if current_action_name == previous_action_name and current_action_reasoning == previous_action_reasoning:
                    # print(f"Skipping {current_action_name} {current_action_reasoning}")
                    continue
                else:
                    # print(f"Adding {current_action_name} {current_action_reasoning}")
                    actual_actions.append(action_json)
                    previous_action_name, previous_action_reasoning = current_action_name, current_action_reasoning
            except:
                try:
                    action_json = agent._parse_natural_language_with_llm(response, pure_text=True)
                    current_action_name = action_json['name']
                    current_action_reasoning = action_json['arguments']['reasoning']
                    if current_action_name == previous_action_name and current_action_reasoning == previous_action_reasoning:
                        #   print(f"Skipping {current_action_name} {current_action_reasoning}")
                        continue
                    else:
                        # print(f"Adding {current_action_name} {current_action_reasoning}")
                        actual_actions.append(action_json)
                        previous_action_name, previous_action_reasoning = current_action_name, current_action_reasoning
                except:
                    print(f"Error parsing action: {response}")
                    continue
            
        if len(actual_actions) < 3:
            continue
        actions_desc = f"EXAMPLE: {task_description}\n"
        for action in actual_actions:
            actions_desc += f"{action['name']}: {action['arguments']['reasoning']}\n"
        action_texts_list.append(actions_desc)
    
    # print('action_texts_list', action_texts_list[:similar_num])
    if len(action_texts_list) > 0:
        return '\n'.join(action_texts_list[:similar_num])
    else:
        return ""
    
    
def construct_experience_memory_continuous(args, current_question, agent):
    evaluation_type = args.evaluation_type
    domain = args.domain
    similar_num = 3
    selected_conversations = get_selected_conversations(current_question, evaluation_type, domain, similar_num+5)
    action_texts_list = []
    images_list = []
    for conversation_file in selected_conversations:
        with open(conversation_file, 'r') as f:
            config_file = json.loads(f.readline())
            task_description = config_file.get('task_description', '')
            # print(task_description)
            if task_description == '':
                print(f"Task description is empty for {conversation_file}")
                continue
            conversation_list = [json.loads(line) for line in f][1:]
        responses_list = [conversation['response'] for conversation in conversation_list]
        images_list_per_conversation = []
        for conversation in conversation_list:
            messages = conversation['messages']
            for message in messages:
                if isinstance(message['content'], list):
                    for item in message['content']:
                        if item['type'] == 'image_url':
                            image_url = item['image_url']['url']
                            image_bytes = base64.b64decode(image_url.split(',')[1])
                            image = Image.open(BytesIO(image_bytes))
                            images_list_per_conversation.append(image)
        if len(images_list_per_conversation) != len(responses_list):
            print(f"Error: {conversation_file} has {len(images_list_per_conversation)} images and {len(responses_list)} responses")
            continue
        images_list.append(images_list_per_conversation)
        
        
        actual_actions = []
        actual_images = []
        previous_action_name, previous_action_reasoning = None, None
        # previous_action_str = ""
        for response, image in zip(responses_list, images_list_per_conversation):
            if isinstance(response, list):
                response = response[0]
            if 'content' in response:
                response = response['content']
            action_json, current_action_name, current_action_reasoning = self.parse_action_from_response(response, image)
            if action_json:
                if current_action_name == previous_action_name and current_action_reasoning == previous_action_reasoning:
                    continue
                else:
                    actual_actions.append(action_json)
                    actual_images.append(image)
                    previous_action_name, previous_action_reasoning = current_action_name, current_action_reasoning
            
        if len(actual_actions) < 3:
            continue
        acions_desc = f"EXAMPLE: {task_description}\n"
        for action in actual_actions:
            acions_desc += f"{action['name']}: {action['arguments']['reasoning']}\n"
        action_texts_list.append(acions_desc)
        images_list.append(actual_images)
    # print('action_texts_list', action_texts_list)
    if len(action_texts_list) > 0:
        return action_texts_list[:similar_num], images_list[:similar_num]
    else:
        return [], []


class Memory:
    """
    A new Memory class that directly uses all jsonl files from /data/wenyi/training_data
    and creates a single pool of memories with embeddings for better generalization.
    """
    
    def __init__(self, training_data_path="training_data/", agent=None, faiss_index_path=None, multimodal=False):
        self.training_data_path = training_data_path
        self.multimodal = multimodal
        self.selected_conversations = None
        self.agent = agent
        if multimodal:
            self.clip_similarity = CLIPMultimodalSimilarity()
        else:
            self.clip_similarity = CLIPTextSimilarity()
            
        self.memories = []  # Single pool of all memories
        self.embeddings = None  # Embedding matrix for all memories
        self.faiss_index = None  # FAISS index for fast similarity search

        if faiss_index_path is None:
            self._load_all_conversations()
            self._create_faiss_index()
            self.save_index(f"/home/wenyi/GUI-Agent/memory/memory_index/{'multimodal' if multimodal else 'text'}_{datetime.now().strftime('m-%d_%H-%M')}")
        else:
            self.load_index(faiss_index_path)
    
    def _load_all_conversations(self):
        """Load all conversations from the training data directory into a single pool."""
        print("Loading all conversations from training data...")
        print(f"[Memory] Scanning for logs in: {self.training_data_path}")
        
        # Get all dataset folders
        dataset_folders = [f for f in os.listdir(self.training_data_path) 
                          if os.path.isdir(os.path.join(self.training_data_path, f))]
        seen_files = set()
        total_conversations = 0

        # --- Existing nested structure loading ---
        for dataset in dataset_folders:
            dataset_path = os.path.join(self.training_data_path, dataset)
            
            # Get all domain folders within each dataset
            domain_folders = [f for f in os.listdir(dataset_path) 
                             if os.path.isdir(os.path.join(dataset_path, f))]
            
            for domain in domain_folders:
                domain_path = os.path.join(dataset_path, domain)
                
                # Get all model folders (like qwen2.5-vl)
                model_folders = [f for f in os.listdir(domain_path) 
                                if os.path.isdir(os.path.join(domain_path, f))]
                
                for model in model_folders:
                    model_seen_files = set()
                    model_path = os.path.join(domain_path, model)
                    
                    # Get all test folders or direct jsonl files
                    items = os.listdir(model_path)
                    
                    # Handle different structures: some have test folders, others have direct jsonl files
                    jsonl_files = []
                    for item in items:
                        item_path = os.path.join(model_path, item)
                        if os.path.isfile(item_path) and item.endswith('.jsonl'):
                            if item_path not in model_seen_files:
                                jsonl_files.append(item_path)
                                model_seen_files.add(item_path)
                        elif os.path.isdir(item_path):
                            # Check if this directory contains jsonl files
                            sub_items = os.listdir(item_path)
                            for sub_item in sub_items:
                                sub_item_path = os.path.join(item_path, sub_item)
                                if os.path.isfile(sub_item_path) and sub_item.endswith('.jsonl'):
                                    if sub_item_path not in model_seen_files:
                                        jsonl_files.append(sub_item_path)
                                        model_seen_files.add(sub_item_path)
                    
                    # Load conversations from jsonl files
                    for jsonl_file in jsonl_files:
                        try:
                            with open(jsonl_file, 'r') as f:
                                lines = f.readlines()
                                if len(lines) < 2:
                                    continue
                                
                                # Read the first line to get task description
                                first_line = lines[0].strip()
                                if first_line:
                                    config = json.loads(first_line)
                                    task_description = config.get('task_description', '')
                                    
                                    if task_description:
                                        # Create prefixed query for better retrieval
                                        prefixed_query = f"{model}_{dataset}_{domain}: {task_description}"
                                        if prefixed_query in seen_files:
                                            continue
                                        print('task_description: ', task_description)
                                        
                                        # Extract image from second line if multimodal
                                        base64_image = None
                                        if self.multimodal and len(lines) >= 2:
                                            try:
                                                second_line = lines[1].strip()
                                                if second_line:
                                                    second_data = json.loads(second_line)
                                                    # Extract base64 image from the second line
                                                    base64_image = self._extract_base64_image(second_data)
                                            except Exception as e:
                                                print(f"Error extracting image from {jsonl_file}: {e}")
                                        
                                        # check whether the task list is valid
                                        conversation_list = [json.loads(line) for line in lines][1:]
                                        responses_list = [conversation['response'] for conversation in conversation_list]
                                        print('responses_list: ', len(responses_list))
                                        actual_actions = []
                                        previous_action_name, previous_action_reasoning = None, None
                                        for response in responses_list:
                                            action_json, current_action_name, current_action_reasoning = self.parse_action_from_response(response)
                                            if action_json:
                                                if current_action_name == previous_action_name and current_action_reasoning == previous_action_reasoning:
                                                    continue
                                                else:
                                                    actual_actions.append(action_json)
                                                    previous_action_name, previous_action_reasoning = current_action_name, current_action_reasoning
                                            else:
                                                print(f"Error parsing action: {response}")
                                                continue
                                        # After parsing, if skipping:
                                        if len(actual_actions) < 3:
                                            print(f"[Memory] Skipping {jsonl_file}: not enough actions.")
                                            continue
                                        # If adding:
                                        print(f"[Memory] Loaded conversation from {jsonl_file}")
                                        self.memories.append({
                                            'file_path': jsonl_file,
                                            'task_description': task_description,
                                            'prefixed_query': prefixed_query,
                                            'dataset': dataset,
                                            'domain': domain,
                                            'base64_image': base64_image
                                        })
                                        seen_files.add(prefixed_query)
                                        total_conversations += 1
                                        # if total_conversations  >= 10:
                                        #     return
                        
                        except Exception as e:
                            print(f"Error loading {jsonl_file}: {e}")
                            continue

        # --- NEW: Scan flat training_data/ directory for .jsonl files ---
        print(f"[Memory] Scanning for conversation logs in: {self.training_data_path}")

        flat_jsonl_files = [
            os.path.join(self.training_data_path, f)
            for f in os.listdir(self.training_data_path)
            if f.endswith('.jsonl') and os.path.isfile(os.path.join(self.training_data_path, f))
        ]
        print(f"[Memory] Found {len(flat_jsonl_files)} flat .jsonl files.")

        for jsonl_file in flat_jsonl_files:
            print(f"[Memory] Processing file: {jsonl_file}")
            try:
                with open(jsonl_file, 'r') as f:
                    lines = f.readlines()
                    print(f"[Memory] File {jsonl_file} has {len(lines)} lines.")
                    if len(lines) < 2:
                        continue
                    first_line = lines[0].strip()
                    if first_line:
                        config = json.loads(first_line)
                        task_description = config.get('task_description', '')
                        # Use 'unknown' for missing metadata
                        dataset = config.get('dataset', 'unknown')
                        domain = config.get('domain', 'unknown')
                        model = config.get('model', 'unknown')
                        prefixed_query = f"{model}_{dataset}_{domain}: {task_description}"
                        if prefixed_query in seen_files:
                            continue
                        print('task_description: ', task_description)
                        # Extract image if multimodal
                        base64_image = None
                        if self.multimodal and len(lines) >= 2:
                            try:
                                second_line = lines[1].strip()
                                if second_line:
                                    second_data = json.loads(second_line)
                                    base64_image = self._extract_base64_image(second_data)
                            except Exception as e:
                                print(f"Error extracting image from {jsonl_file}: {e}")
                        # Check actions
                        conversation_list = [json.loads(line) for line in lines][1:]
                        responses_list = [conversation['response'] for conversation in conversation_list]
                        actual_actions = []
                        previous_action_name, previous_action_reasoning = None, None
                        for response in responses_list:
                            action_json, current_action_name, current_action_reasoning = self.parse_action_from_response(response)
                            if action_json:
                                if current_action_name == previous_action_name and current_action_reasoning == previous_action_reasoning:
                                    continue
                                else:
                                    actual_actions.append(action_json)
                                    previous_action_name, previous_action_reasoning = current_action_name, current_action_reasoning
                            else:
                                print(f"Error parsing action: {response}")
                                continue
                        # After parsing, if skipping:
                        if len(actual_actions) < 3:
                            print(f"[Memory] Skipping {jsonl_file}: not enough actions.")
                            continue
                        # If adding:
                        print(f"[Memory] Loaded conversation from {jsonl_file}")
                        self.memories.append({
                            'file_path': jsonl_file,
                            'task_description': task_description,
                            'prefixed_query': prefixed_query,
                            'dataset': dataset,
                            'domain': domain,
                            'base64_image': base64_image
                        })
                        seen_files.add(prefixed_query)
                        total_conversations += 1
            except Exception as e:
                print(f"Error loading {jsonl_file}: {e}")
                continue

        print(f"Total conversations loaded: {len(self.memories)}")
    
    def _extract_base64_image(self, data):
        """Extract base64 image from conversation data."""
        try:
            # Check if data has messages
            if 'messages' in data:
                messages = data['messages']
                for msg in messages:
                    if isinstance(msg.get('content'), list):
                        for item in msg['content']:
                            if item.get('type') == 'image_url':
                                return item['image_url']['url']
            return None
        except Exception as e:
            print(f"Error extracting base64 image: {e}")
            return None
    
    def _create_faiss_index(self):
        """Create FAISS index for fast similarity search."""
        print("Creating FAISS index for all memories...")
        if not self.memories:
            print("No memories to create FAISS index for")
            return
        
        # Extract all prefixed queries and base64 images
        prefixed_queries = [memory['prefixed_query'] for memory in self.memories]
        base64_images = [memory.get('base64_image') for memory in self.memories]
        
        # Create embeddings using CLIP
        if self.multimodal:
            # For multimodal, we always create multimodal embeddings
            # Use None for missing images to maintain consistent dimensions
            self.embeddings = self.clip_similarity.get_multimodal_embeddings(prefixed_queries, base64_images)
        else:
            self.embeddings = self.clip_similarity.get_text_embeddings(prefixed_queries)
        
        print(f"Created embeddings matrix with shape: {self.embeddings.shape}")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # Add vectors to the index
        self.faiss_index.add(self.embeddings.astype('float32'))
        
        print(f"Created FAISS index with {self.faiss_index.ntotal} vectors")
    
    def retrieve_similar_conversations(self, current_question, current_image=None, model=None, similar_num=3):
        """
        Retrieve similar conversations based on text and/or image similarity from the single memory pool using FAISS.
        
        Args:
            current_question: The current query/question
            current_image: Optional base64 encoded image for multimodal search
            similar_num: Number of similar conversations to retrieve
        
        Returns:
            List of selected conversation file paths
        """
        if not self.memories or self.faiss_index is None:
            print("No memories available for retrieval")
            return []
        if model is not None:
            current_question = f"{model}: {current_question}"
        # Get embedding for current question and image
        if self.multimodal:
            if current_image is not None:
                current_embedding = self.clip_similarity.get_multimodal_embeddings([current_question], [current_image])
            else:
                # For multimodal mode with no image, we need to create embeddings with the same dimension
                # as the stored embeddings (which are text+image concatenated)
                text_embedding = self.clip_similarity.get_text_embeddings([current_question])
                # Create zero embeddings for the image part to match the stored dimension
                zero_image_embedding = np.zeros_like(text_embedding)
                current_embedding = np.concatenate([text_embedding, zero_image_embedding], axis=1)
        else:
            current_embedding = self.clip_similarity.get_text_embeddings([current_question])
            zero_image_embedding = np.zeros_like(current_embedding)
            current_embedding = np.concatenate([current_embedding, zero_image_embedding], axis=1)
        
        # Normalize embedding for cosine similarity
        faiss.normalize_L2(current_embedding)
        
        # Search using FAISS
        similarities, indices = self.faiss_index.search(
            current_embedding.astype('float32'), similar_num
        )
        
        selected_conversations = []
        for i, (score, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx != -1:  # FAISS returns -1 for invalid indices
                selected_conversations.append(self.memories[idx]['file_path'])
                print(f"Score: {score:.4f} - {self.memories[idx]['prefixed_query']}")
        
        return selected_conversations
    
    def parse_action_from_response(self, response, image=None):
        """
        Parse action from response with fallback to LLM parsing.
        
        Args:
            response: The response to parse
            agent: The agent instance for LLM parsing fallback
            image: Optional image for context (used in continuous memory)
        
        Returns:
            tuple: (action_json, current_action_name, current_action_reasoning) or (None, None, None) if parsing fails
        """
        try:
            if isinstance(response, list):
                response = response[0]
            if isinstance(response, dict) and 'content' in response:
                response = response['content']
            action_json = parse_action_json(response).get('function_call', {})
            
            if 'name' in action_json:
                current_action_name = action_json['name']
                current_action_reasoning = action_json['arguments']['reasoning']
            elif 'action' in action_json:
                current_action_name = action_json['action']
                current_action_reasoning = action_json['reasoning']
            elif 'action_type' in action_json:
                current_action_name = action_json['action_type']
                current_action_reasoning = action_json['reasoning']
            elif 'type' in action_json:
                current_action_name = action_json['type']
                current_action_reasoning = action_json['reasoning']
            elif isinstance(list(action_json.values())[0], dict):
                current_action_name = list(action_json.keys())[0]
                current_action_reasoning = list(action_json.values())[0]['reasoning']
            else:
                print(f"Error: {action_json} has no name, action, or action_type")
                return None, None, None
            
            action_json['name'] = current_action_name
            action_json['arguments'] = {'reasoning': current_action_reasoning}
            
            return action_json, current_action_name, current_action_reasoning
        
        except:
            with open(f"/home/wenyi/GUI-Agent/memory/error_responses.txt", 'a') as f:
                f.write('*'*50 + '\n' + str(response) + '\n' + '*'*50 + '\n')
                f.close()
            try:
                action_json = self.agent._parse_natural_language_with_llm(response, pure_text=True)
                current_action_name = action_json['name']
                current_action_reasoning = action_json['arguments']['reasoning']
                
                return action_json, current_action_name, current_action_reasoning
            except:
                print(f"Error parsing action: {response}")
                return None, None, None

    def construct_experience_memory(self, current_question, agent, current_image=None, dataset=None, domain=None, similar_num=3):
        """
        Construct experience memory from similar conversations.
        
        Args:
            current_question: The current query/question
            agent: The agent instance for parsing actions
            current_image: Optional base64 encoded image for multimodal search
            dataset: Optional dataset filter
            domain: Optional domain filter
            similar_num: Number of similar conversations to use
        
        Returns:
            Formatted experience memory string
        """
        current_question = f"{dataset}_{domain}: {current_question}" if dataset and domain else current_question
        selected_conversations = self.retrieve_similar_conversations(
            current_question=current_question, current_image=current_image, similar_num=similar_num + 5
        )
        self.selected_conversations = selected_conversations
        
        action_texts_list = []
        images_list = []
        
        for conversation_file in selected_conversations:
            try:
                with open(conversation_file, 'r') as f:
                    config_file = json.loads(f.readline())
                    task_description = config_file.get('task_description', '')
                    
                    if task_description == '':
                        print(f"Task description is empty for {conversation_file}")
                        continue
                    
                    conversation_list = [json.loads(line) for line in f][1:]
                
                responses_list = [conversation['response'] for conversation in conversation_list]
                images_list_per_conversation = []
                for conversation in conversation_list:
                    messages = conversation['messages']
                    for message in messages:
                        if isinstance(message['content'], list):
                            for item in message['content']:
                                if item['type'] == 'image_url':
                                    image_url = item['image_url']['url']
                                    image_bytes = base64.b64decode(image_url.split(',')[1])
                                    image = Image.open(BytesIO(image_bytes))
                                    images_list_per_conversation.append(image)
                if len(images_list_per_conversation) != len(responses_list):
                    print(f"Error: {conversation_file} has {len(images_list_per_conversation)} images and {len(responses_list)} responses")
                    continue
                
                actual_actions = []
                actual_images = []
                previous_action_name, previous_action_reasoning = None, None
                
                for response, image in zip(responses_list, images_list_per_conversation):
                    if isinstance(response, list):
                        response = response[0]
                    if 'content' in response:
                        response = response['content']
                    
                    action_json, current_action_name, current_action_reasoning = self.parse_action_from_response(response, image)
                    
                    if action_json:
                        if current_action_name == previous_action_name and current_action_reasoning == previous_action_reasoning:
                            continue
                        else:
                            actual_actions.append(action_json)
                            actual_images.append(image)
                            previous_action_name, previous_action_reasoning = current_action_name, current_action_reasoning
                    
                    else:
                        print(f"Error parsing action: {response}")
                        continue
                
                if len(actual_actions) < similar_num:
                    continue
                
                actions_desc = f"EXAMPLE: {task_description}\n"
                for action in actual_actions:
                    actions_desc += f"{action['name']}: {action['arguments']['reasoning']}\n"
                
                action_texts_list.append(actions_desc)
                images_list.append(actual_images)
                print('action_texts_list: ', len(action_texts_list))
                print('images_list: ', len(images_list))
                
            except Exception as e:
                print(f"Error processing {conversation_file}: {e}")
                continue
        
        print('*'*50, 'action_texts_list', '*'*50)
        print(len(action_texts_list))
        print(len(action_texts_list))
        print('*'*50, 'action_texts_list', '*'*50)
        
        if len(action_texts_list) > 0:
            return '\n'.join(action_texts_list[:similar_num]), action_texts_list[:similar_num], images_list[:similar_num]
        else:
            return "", [], []
        
    
    def get_available_datasets_and_domains(self):
        """Get list of available datasets and domains."""
        result = {}
        for memory in self.memories:
            dataset = memory['dataset']
            domain = memory['domain']
            if dataset not in result:
                result[dataset] = []
            if domain not in result[dataset]:
                result[dataset].append(domain)
        return result
    
    def save_index(self, filepath):
        """Save the FAISS index, embeddings, and memory data to disk."""
        if self.faiss_index is None:
            print("No FAISS index to save")
            return
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, f"{filepath}.faiss")
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(f"{filepath}.embeddings.npy", self.embeddings)
        
        # Save memory data
        memory_data = {
            'memories': self.memories,
            'embeddings_shape': self.embeddings.shape if self.embeddings is not None else None
        }
        
        with open(f"{filepath}.json", 'w') as f:
            json.dump(memory_data, f, indent=2)
        
        print(f"Saved FAISS index, embeddings, and memory data to {filepath}")
    
    def load_index(self, filepath):
        """Load the FAISS index, embeddings, and memory data from disk."""
        try:
            # Load FAISS index
            self.faiss_index = faiss.read_index(f"{filepath}.faiss")
            
            # Load embeddings
            embeddings_path = f"{filepath}.embeddings.npy"
            if os.path.exists(embeddings_path):
                self.embeddings = np.load(embeddings_path)
                print(f"Loaded embeddings with shape: {self.embeddings.shape}")
            else:
                print("Embeddings file not found, reconstructing from FAISS index...")
                self.embeddings = self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal)
            
            # Load memory data
            with open(f"{filepath}.json", 'r') as f:
                memory_data = json.load(f)
            
            self.memories = memory_data['memories']
            
            print(f"Loaded FAISS index and memory data from {filepath}")
            print(f"Index contains {self.faiss_index.ntotal} vectors")
            print(f"Loaded {len(self.memories)} memories")
            
        except Exception as e:
            print(f"Error loading index from {filepath}: {e}")
            print("Falling back to creating new index...")
            self._load_all_conversations()
            self._create_faiss_index()


    def retrieve_similar_conversations_with_filter(self, current_question, current_image=None, dataset=None, domain=None, similar_num=3):
        """
        Retrieve similar conversations with optional dataset/domain filtering.
        This method filters the memory pool before similarity search.
        
        Args:
            current_question: The current query/question
            current_image: Optional base64 encoded image for multimodal search
            dataset: Optional dataset filter
            domain: Optional domain filter
            similar_num: Number of similar conversations to retrieve
        
        Returns:
            List of selected conversation file paths
        """
        if not self.memories or self.faiss_index is None:
            print("No memories available for retrieval")
            return []
        
        # Filter memories based on dataset and domain if specified
        filtered_memories = []
        filtered_indices = []
        
        for i, memory in enumerate(self.memories):
            if dataset and memory['dataset'] != dataset:
                continue
            if domain and memory['domain'] != domain:
                continue
            filtered_memories.append(memory)
            filtered_indices.append(i)
        
        if not filtered_memories:
            print("No memories found for the specified criteria")
            return []
        
        # Get embeddings for filtered memories
        filtered_embeddings = self.embeddings[filtered_indices]
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(filtered_embeddings)
        
        # Create a temporary FAISS index for filtered embeddings
        dimension = filtered_embeddings.shape[1]
        temp_index = faiss.IndexFlatIP(dimension)
        temp_index.add(filtered_embeddings.astype('float32'))
        
        # Get embedding for current question and image
        if self.multimodal:
            if current_image is not None:
                current_embedding = self.clip_similarity.get_multimodal_embeddings([current_question], [current_image])
            else:
                # For multimodal mode with no image, we need to create embeddings with the same dimension
                # as the stored embeddings (which are text+image concatenated)
                text_embedding = self.clip_similarity.get_text_embeddings([current_question])
                # Create zero embeddings for the image part to match the stored dimension
                zero_image_embedding = np.zeros_like(text_embedding)
                current_embedding = np.concatenate([text_embedding, zero_image_embedding], axis=1)
        else:
            current_embedding = self.clip_similarity.get_text_embeddings([current_question])
        
        # Normalize embedding for cosine similarity
        faiss.normalize_L2(current_embedding)
        
        # Search using temporary FAISS index
        similarities, indices = temp_index.search(
            current_embedding.astype('float32'), similar_num
        )
        
        selected_conversations = []
        for i, (score, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx != -1:  # FAISS returns -1 for invalid indices
                memory_idx = filtered_indices[idx]
                selected_conversations.append(self.memories[memory_idx]['file_path'])
                print(f"Score: {score:.4f} - {self.memories[memory_idx]['prefixed_query']}")
        
        return selected_conversations

    def construct_experience_memory_continuous(self, dataset, domain, current_question, agent, current_image=None, similar_num=10):
        current_question = f"{dataset}_{domain}: {current_question}" if dataset and domain else current_question
        selected_conversations = self.retrieve_similar_conversations(
            current_question=current_question, current_image=current_image, similar_num=similar_num + 5
        )
        
        action_texts_list = []
        images_list = []
        for conversation_file in selected_conversations:
            with open(conversation_file, 'r') as f:
                config_file = json.loads(f.readline())
                task_description = config_file.get('task_description', '')
                # print(task_description)
                if task_description == '':
                    print(f"Task description is empty for {conversation_file}")
                    continue
                conversation_list = [json.loads(line) for line in f][1:]
            responses_list = [conversation['response'] for conversation in conversation_list]
            images_list_per_conversation = []
            for conversation in conversation_list:
                messages = conversation['messages']
                for message in messages:
                    if isinstance(message['content'], list):
                        for item in message['content']:
                            if item['type'] == 'image_url':
                                image_url = item['image_url']['url']
                                image_bytes = base64.b64decode(image_url.split(',')[1])
                                image = Image.open(BytesIO(image_bytes))
                                images_list_per_conversation.append(image)
            if len(images_list_per_conversation) != len(responses_list):
                print(f"Error: {conversation_file} has {len(images_list_per_conversation)} images and {len(responses_list)} responses")
                continue
            images_list.append(images_list_per_conversation)
            
            
            actual_actions = []
            actual_images = []
            previous_action_name, previous_action_reasoning = None, None
            # previous_action_str = ""
            for response, image in zip(responses_list, images_list_per_conversation):
                if isinstance(response, list):
                    response = response[0]
                if 'content' in response:
                    response = response['content']
                action_json, current_action_name, current_action_reasoning = self.parse_action_from_response(response, agent, image)
                if action_json:
                    if current_action_name == previous_action_name and current_action_reasoning == previous_action_reasoning:
                        continue
                    else:
                        actual_actions.append(action_json)
                        actual_images.append(image)
                        previous_action_name, previous_action_reasoning = current_action_name, current_action_reasoning
                else:
                    print(f"Error parsing action: {response}")
                    continue
                
            if len(actual_actions) < similar_num:
                continue
            acions_desc = f"EXAMPLE: {task_description}\n"
            for action in actual_actions:
                acions_desc += f"{action['name']}: {action['arguments']['reasoning']}\n"
            action_texts_list.append(acions_desc)
            images_list.append(actual_images)
            print('action_texts_list: ', len(action_texts_list))
            print('images_list: ', len(images_list))
        # print('action_texts_list', action_texts_list)
        if len(action_texts_list) > 0:
            return action_texts_list[:similar_num], images_list[:similar_num]
        else:
            return [], []
