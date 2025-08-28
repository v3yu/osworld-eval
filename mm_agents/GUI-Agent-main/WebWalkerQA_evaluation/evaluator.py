"""WebWalkerQA evaluation system for GUI Agent"""
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from openai import OpenAI
from browser_env import Action, Trajectory, StateInfo
from .helper_functions import clean_answer, extract_answer_from_config


class WebWalkerQAEvaluator:
    """Evaluator for WebWalkerQA benchmark using LLM fuzzy matching"""
    
    def __init__(self, vllm_client=None, eval_tag: str = "webwalkerqa"):
        self.eval_tag = eval_tag
        self.vllm_client = vllm_client

    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page=None,
        client=None,
    ) -> float:
        """Evaluate WebWalkerQA task using LLM fuzzy matching"""
        
        # Load config
        with open(config_file, "r") as f:
            configs = json.load(f)

        # Get reference answer
        reference_answer = extract_answer_from_config(configs)

        # Get prediction from trajectory
        last_action = self._get_last_action(trajectory)
        pred = clean_answer(last_action.get("answer", ""))
        
        # Use LLM fuzzy matching for evaluation
        score = self._llm_fuzzy_match(pred, reference_answer)
        
        return score

    def _get_last_action(self, trajectory: Trajectory) -> Action:
        """Get the last action from trajectory"""
        if not trajectory or not isinstance(trajectory[-1], dict):
            raise ValueError("The last element of trajectory should be an action")
        return trajectory[-1]

    def _llm_fuzzy_match(self, pred: str, reference: str) -> float:
        """Use LLM for binary yes/no matching, same as MMInA evaluator"""
        try:
            # Initialize OpenAI client for vLLM server
            client = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key="dummy-key"
            )
            
            # Create the prompt for binary matching
            prompt = f"""You are an evaluator that determines if a predicted answer is correct.

Reference Answer: {reference}
Predicted Answer: {pred}

Please evaluate if the predicted answer is correct. Consider:
1. Semantic similarity to the reference answer
2. Key information overlap
3. Factual accuracy

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