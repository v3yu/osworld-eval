"""SuperGPQA evaluation system for GUI Agent"""
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from openai import OpenAI
from browser_env import Action, Trajectory, StateInfo
from .helper_functions import (
    clean_answer, 
    extract_answer_from_config,
    extract_answer_letter_from_config,
    extract_options_from_config,
    get_option_by_letter
)


class SuperGPQAEvaluator:
    """Evaluator for SuperGPQA benchmark using LLM fuzzy matching"""
    
    def __init__(self, vllm_client=None, eval_tag: str = "supergpqa"):
        self.eval_tag = eval_tag
        self.vllm_client = vllm_client

    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page=None,
        client=None,
    ) -> float:
        """Evaluate SuperGPQA task using LLM fuzzy matching"""
        
        # Load config
        with open(config_file, "r") as f:
            configs = json.load(f)

        # Get reference data
        reference_answer = extract_answer_from_config(configs)
        reference_letter = extract_answer_letter_from_config(configs)
        options = extract_options_from_config(configs)

        # Get prediction from trajectory
        last_action = self._get_last_action(trajectory)
        pred = clean_answer(last_action.get("answer", ""))
        
        # Use LLM fuzzy matching for evaluation
        score = self._llm_fuzzy_match(pred, reference_answer, reference_letter, options)
        
        return score

    def _get_last_action(self, trajectory: Trajectory) -> Action:
        """Get the last action from trajectory"""
        if not trajectory or not isinstance(trajectory[-1], dict):
            raise ValueError("The last element of trajectory should be an action")
        return trajectory[-1]

    def _llm_fuzzy_match(self, pred: str, reference: str, reference_letter: str, options: List[str]) -> float:
        """Use LLM for binary yes/no matching, same as MMInA evaluator"""
        try:
            # Initialize OpenAI client for vLLM server
            client = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key="dummy-key"
            )
            
            # Format options for context
            options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            
            # Create the prompt for binary matching
            prompt = f"""You are an evaluator that determines if a predicted answer is correct for a multiple-choice question.

Question Options:
{options_text}

Correct Answer: {reference} (Option {reference_letter})
Predicted Answer: {pred}

Please evaluate if the predicted answer is correct. Consider:
1. Semantic similarity to the correct answer
2. Whether the prediction matches the correct option letter or text
3. Key information overlap
4. Factual accuracy

Respond with only "yes" or "no":
- "yes": if the predicted answer is correct or equivalent to the reference answer
- "no": if the predicted answer is incorrect, incomplete, or irrelevant"""

            # Call the model
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-VL-7B-Instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful evaluator that provides binary yes/no responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=5,
                stop=None
            )
            
            # Extract the response
            answer_text = response.choices[0].message.content.strip().lower()
            
            # Parse the yes/no response
            if 'yes' in answer_text:
                return 1.0
            elif 'no' in answer_text:
                return 0.0
            else:
                print(f"Could not parse yes/no from response: '{answer_text}'")
                return 0.0
                
        except Exception as e:
            print(f"Error in fuzzy matching: {e}")
            return 0.0


class SuperGPQALetterEvaluator:
    """Evaluator that checks if the predicted answer letter matches the correct letter"""
    
    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page=None,
        client=None,
    ) -> float:
        """Evaluate if the predicted answer letter is correct"""
        
        # Load config
        with open(config_file, "r") as f:
            configs = json.load(f)

        # Get reference letter
        reference_letter = extract_answer_letter_from_config(configs)
        
        # Get prediction from trajectory
        last_action = self._get_last_action(trajectory)
        pred = clean_answer(last_action.get("answer", ""))
        
        # Extract letter from prediction (look for A, B, C, D, etc.)
        pred_letter = self._extract_letter_from_prediction(pred)
        
        # Compare letters
        if pred_letter and pred_letter.upper() == reference_letter.upper():
            return 1.0
        else:
            return 0.0

    def _get_last_action(self, trajectory: Trajectory) -> Action:
        """Get the last action from trajectory"""
        if not trajectory or not isinstance(trajectory[-1], dict):
            raise ValueError("The last element of trajectory should be an action")
        return trajectory[-1]

    def _extract_letter_from_prediction(self, pred: str) -> str:
        """Extract answer letter from prediction text"""
        if not pred:
            return ""
        
        # Look for single letters (A, B, C, D, etc.)
        import re
        letter_match = re.search(r'\b[A-Z]\b', pred.upper())
        if letter_match:
            return letter_match.group()
        
        return ""


class SuperGPQACombinedEvaluator:
    """Combined evaluator that checks both semantic correctness and letter accuracy"""
    
    def __init__(self, vllm_client=None, eval_tag: str = "supergpqa_combined"):
        self.eval_tag = eval_tag
        self.vllm_client = vllm_client
        self.semantic_evaluator = SuperGPQAEvaluator(vllm_client, "supergpqa_semantic")
        self.letter_evaluator = SuperGPQALetterEvaluator()

    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page=None,
        client=None,
    ) -> float:
        """Evaluate both semantic correctness and letter accuracy"""
        
        # Evaluate semantic correctness
        semantic_score = self.semantic_evaluator(trajectory, config_file, page, client)
        
        # Evaluate letter accuracy
        letter_score = self.letter_evaluator(trajectory, config_file, page, client)
        
        # Combined score (both must be correct)
        combined_score = semantic_score * letter_score
        
        return combined_score 