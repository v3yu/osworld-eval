"""Test runner for SuperGPQA evaluation"""
import argparse
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from agent.agent import FunctionCallAgent
from .evaluator import SuperGPQAEvaluator, SuperGPQALetterEvaluator, SuperGPQACombinedEvaluator
from .helper_functions import (
    load_supergpqa_data,
    create_supergpqa_config,
    save_supergpqa_config,
    validate_supergpqa_data,
    format_question_with_options
)
from utils.early_stop import early_stop
from utils.help_functions import save_scores_to_json


class SuperGPQATestRunner:
    """Test runner for SuperGPQA evaluation"""
    
    def __init__(self, args: argparse.Namespace, agent: FunctionCallAgent):
        self.args = args
        self.agent = agent
        self.logger = logging.getLogger("logger")
        
        # Initialize environment
        self.env = ScriptBrowserEnv(
            headless=True,
            slow_mo=args.slow_mo,
            viewport_size={
                "width": args.viewport_width,
                "height": args.viewport_height,
            },
            save_trace_enabled=args.save_trace_enabled,
            sleep_after_execution=args.sleep_after_execution,
            args=args,
        )
        
        # Initialize tracking variables
        self.scores = {}
        self.trajSOM = {}
        self.trajImages = {}
        self.trajActions = {}
        self.trajSuccess = {}
        self.metrics_dict = {}
        self.results = []
        
        # Load existing data if saving example memory
        if args.save_examples_memory:
            self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing data for memory saving"""
        metrics_json_path = f"{self.args.result_dir}/metrics.json"
        if os.path.exists(metrics_json_path):
            with open(metrics_json_path) as json_file:
                self.metrics_dict = json.load(json_file)

        states_path = f"{self.args.result_dir}/states.pkl"
        if os.path.exists(states_path):
            with open(states_path, "rb") as f:
                self.trajSOM = pickle.load(f)

        actions_path = f"{self.args.result_dir}/actions.pkl"
        if os.path.exists(actions_path):
            with open(actions_path, "rb") as f:
                self.trajActions = pickle.load(f)

        success_path = f"{self.args.result_dir}/success.pkl"
        if os.path.exists(success_path):
            with open(success_path, "rb") as f:
                self.trajSuccess = pickle.load(f)
    
    def run(self, config_file_list: list[str] = None):
        """Run the main test loop"""
        # Load SuperGPQA data
        df = self.load_supergpqa_data()
        test_df = df.head(self.args.max_samples)
        
        self.logger.info(f"Starting SuperGPQA evaluation with {len(test_df)} tests")
        
        # Process each question
        for i, (_, row) in enumerate(test_df.iterrows()):
            config = self.create_test_config(row, i + 1)
            config_file = f"{self.args.result_dir}/config/config_{config['task_id']}.json"
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            save_supergpqa_config(config, Path(config_file))
            
            self._process_config_file(config_file, row)
        
        # Save results
        self._save_results()
        
        # Close environment
        self.env.close()
    
    def load_supergpqa_data(self, data_path: str = None) -> pd.DataFrame:
        """Load SuperGPQA dataset"""
        if data_path is None:
            data_path = "hf://datasets/m-a-p/SuperGPQA/SuperGPQA-all.jsonl"
        
        self.logger.info(f"Loading SuperGPQA data from {data_path}...")
        df = load_supergpqa_data(data_path)
        
        if df.empty:
            self.logger.error("Failed to load SuperGPQA data. Make sure you're logged in with `huggingface-cli login`")
            return df
        
        self.logger.info(f"Loaded {len(df)} SuperGPQA questions")
        
        # # Filter valid questions
        # valid_df = df[df.apply(validate_supergpqa_data, axis=1)]
        # self.logger.info(f"Found {len(valid_df)} valid questions")
        
        return df
    
    def create_test_config(self, row: pd.Series, test_id: int) -> Dict[str, Any]:
        """Create test config from SuperGPQA data row"""
        
        # Format the question with options
        question_text = format_question_with_options(row["question"], row["options"])
        
        # Create the full prompt
        prompt = f"""You are a helpful Web Search Agent that answers multiple-choice questions. 
You are given a question and a list of options.
You can search the web for information and use provided tools to get the answer.
Please provide your final answer with only the reasoning and <answer>A/B/C/D/.......</answer>
{question_text}
"""

        config = create_supergpqa_config(
            question=row["question"],
            options=row["options"],
            answer=row["answer"],
            answer_letter=row["answer_letter"],
            discipline=row.get("discipline", ""),
            field=row.get("field", ""),
            subfield=row.get("subfield", ""),
            difficulty=row.get("difficulty", ""),
            is_calculation=row.get("is_calculation", False),
            task_id=row.get("uuid", f"test_{test_id}"),
            start_url="https://www.bing.com"
        )
        
        # Add the prompt to config
        config["prompt"] = prompt
        
        return config
    
    def _process_config_file(self, config_file: str, row: pd.Series):
        """Process a single SuperGPQA config file"""
        discipline = row.get("discipline", "Unknown")
        difficulty = row.get("difficulty", "Unknown")
        
        # Initialize scores for this discipline
        if discipline not in self.scores:
            self.scores[discipline] = {}
            self.scores[discipline]['task_scores'] = []
            self.scores[discipline]['difficulty_scores'] = {}
        
        action_list = []
        render_helper = RenderHelper(config_file, self.args.result_dir)

        # Get task info
        with open(config_file) as f:
            config = json.load(f)
            question = config["question"]
            prompt = config["prompt"]
            task_id = config.get("task_id", "unknown")
        
        episode_id = f"{discipline}_{task_id}"
        
        self.logger.info(f"[Config file]: {config_file}")
        self.logger.info(f"[Question]: {question[:100]}...")
        self.logger.info(f"[Discipline]: {discipline}")
        self.logger.info(f"[Difficulty]: {difficulty}")
        
        self.agent.reset(config_file)
        trajectory: Trajectory = []
        # Start conversation for this task if training data collection is enabled
        if hasattr(self.agent, 'training_collector') and self.agent.training_collector:
            from utils.training_data_collector import get_collector
            collector = get_collector()
            if collector and collector.enabled:
                # Create conversation ID from task info
                conversation_id = task_id
                collector.start_conversation(
                    conversation_id=conversation_id,
                    task_description=prompt
                )
                self.logger.info(f"Started conversation collection for task: {conversation_id}")
        
        # Environment reset
        obs, info = self.env.reset(options={"config_file": config_file})
        current_url = info["page"].url
        state_info: StateInfo = {"observation": obs, "info": info, "current_url": current_url}
        trajectory.append(state_info)
        print("CURRENT: ", current_url)
        
        meta_data = {"action_history": [], "url": current_url}
        
        count = 0
        
        # Process the question
        while True:
            current_url = current_url.lower()
            meta_data["url"] = current_url

            early_stop_flag, stop_info = early_stop(
                trajectory, self.args.max_steps, {
                    "parsing_failure": self.args.parsing_failure_th,
                    "repeating_action": self.args.repeating_action_failure_th,
                }
            )

            if early_stop_flag:
                if self.args.enable_fallback:
                    from utils.fallback_answer import generate_fallback_answer                       
                    self.logger.info(f"[Fallback] Early stop detected: {stop_info}. Generating fallback answer...")
                    
                    # Generate fallback answer using the LLM
                    fallback_result = generate_fallback_answer(
                        question=prompt,
                        trajectory=trajectory,
                        model=self.agent.llm,
                        num_screenshots=self.args.fallback_screenshots
                    )
                    # Create a stop action with the fallback answer
                    action = create_stop_action(fallback_result['answer'])
                    # Integrate the fallback answer into the trajectory
                    trajectory.append(action)
                    self.logger.info(f"[Fallback] Generated fallback answer: {fallback_result['answer']}...")
                else:
                    action = create_stop_action(f"Early stop: {stop_info}")
            else:
                def gen_action(intent, meta, error_message=None):
                    if error_message:
                        intent = f"{intent}\n[ERROR_FEEDBACK]: {error_message}"
                    action, meta =  self.agent.next_action_custom(
                        trajectory,
                        intent,
                        meta_data=meta,
                        model=self.args.loaded_model,
                        args=self.args,
                    )
                    return action, meta
                
                if self.args.action_check:
                    from utils.action_check import action_self_check
                    action, meta_data = action_self_check(gen_action, prompt, self.env.page, trajectory, max_retries=3, repeat_threshold=self.args.repeating_action_failure_th)
                else:
                    action, meta_data = gen_action(prompt, meta_data)
                
            if isinstance(action, list):
                trajectory.extend(action)
            else:
                trajectory.append(action)
            
            action_str = get_action_description(action)
            render_helper.render(
                action, state_info, meta_data, self.args.render_screenshot
            )
            meta_data["action_history"].append(action_str)
            meta_data["page"] = self.env.page

            if isinstance(action, list):
                last_action_type = action[-1]["action_type"]
            else:
                last_action_type = action["action_type"]
            
            if last_action_type in [ActionTypes.STOP, 'finished']:
                self.logger.info(f"[Question completed]")
                break
            
            obs, _, terminated, _, info, current_url = self.env.step(action, observation=obs)
            print("CURRENT: ", current_url)

            state_info = {"observation": obs, "info": info}
            trajectory.append(state_info)

            if terminated:
                trajectory.append(create_stop_action(""))
                self.logger.info(f"[Question terminated]")
                break
            
            count += 1

        # Store trajectory info
        self.trajActions[config_file] = action_list

        # Evaluate the scores
        evaluator = SuperGPQAEvaluator()
        score = evaluator(
            trajectory=trajectory,
            config_file=config_file,
            page=self.env.page,
            client=self.env.get_page_client(self.env.page),
        )
        
        self.metrics_dict[config_file] = {
            "config": config_file,
            "success": score,
            "discipline": discipline,
            "difficulty": difficulty,
        }
        self.trajSuccess[config_file] = score
        
        # Update scores
        self.scores[discipline]['task_scores'].append(int(score == 1))
        self.scores[discipline]['task_success_rate'] = sum(self.scores[discipline]['task_scores']) / len(self.scores[discipline]['task_scores'])
        
        # Update difficulty scores
        if difficulty not in self.scores[discipline]['difficulty_scores']:
            self.scores[discipline]['difficulty_scores'][difficulty] = []
        self.scores[discipline]['difficulty_scores'][difficulty].append(int(score == 1))
        
        result = "PASS" if score == 1 else "FAIL"
        self.logger.info(f"[Result] ({result}) {config_file}")
        self.logger.info(f"[Result] Current {discipline} success rate: {self.scores[discipline]['task_success_rate']}")
        
        # Save memory and metrics if required
        if self.args.save_examples_memory:
            self.logger.info("Saving memory files...")
        
            with open(f"{self.args.result_dir}/states.pkl", "wb") as f:
                pickle.dump(self.trajSOM, f)

            with open(f"{self.args.result_dir}/actions.pkl", "wb") as f:
                pickle.dump(self.trajActions, f)

            with open(f"{self.args.result_dir}/success.pkl", "wb") as f:
                pickle.dump(self.trajSuccess, f)

            with open(f"{self.args.result_dir}/metrics.json", "w") as outfile:
                json.dump(self.metrics_dict, outfile, indent=4, sort_keys=True)
        
        render_helper.close()
        
        # End conversation for this task if training data collection is enabled
        if hasattr(self.agent, 'training_collector') and self.agent.training_collector:
            from utils.training_data_collector import get_collector
            collector = get_collector()
            if collector and collector.enabled and collector.current_conversation_id:
                # Create conversation summary
                conversation_summary = {
                    "task_id": config_file.split('/')[-1].split('.')[0],
                    "success": score,
                    "final_url": current_url,
                    "task_completed": True,
                    "task_description": prompt
                }
                
                # End the conversation
                saved_file = collector.end_conversation(conversation_summary)
                if saved_file:
                    self.logger.info(f"Conversation saved: {saved_file}")
    
    def _save_results(self):
        """Save final results"""
        with open(Path(self.args.result_dir) / "scores.json", "w") as f:
            json.dump(self.scores, f, indent=4)
        
        # Save detailed results
        results_file = Path(self.args.result_dir) / "supergpqa_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # # Also save as CSV for easy analysis
        # if self.results:
        #     df_results = pd.DataFrame(self.results)
        #     csv_file = Path(self.args.result_dir) / "supergpqa_results.csv"
        #     df_results.to_csv(csv_file, index=False)
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print evaluation summary"""
        if not self.scores:
            self.logger.info("No results to summarize")
            return
        
        self.logger.info(f"\n=== SuperGPQA Evaluation Summary ===")
        
        total_tests = sum(len(discipline_data['task_scores']) for discipline_data in self.scores.values())
        total_success = sum(sum(discipline_data['task_scores']) for discipline_data in self.scores.values())
        
        self.logger.info(f"Total tests: {total_tests}")
        self.logger.info(f"Total successful tests: {total_success}")
        self.logger.info(f"Overall success rate: {total_success/total_tests*100:.1f}%")
        
        # Results by discipline
        for discipline, discipline_data in self.scores.items():
            success_rate = discipline_data['task_success_rate']
            test_count = len(discipline_data['task_scores'])
            self.logger.info(f"  {discipline}: {test_count} tests, success rate: {success_rate:.3f}")
            
            # Results by difficulty within discipline
            if 'difficulty_scores' in discipline_data:
                for difficulty, scores in discipline_data['difficulty_scores'].items():
                    difficulty_rate = sum(scores) / len(scores)
                    self.logger.info(f"    {difficulty}: {len(scores)} tests, success rate: {difficulty_rate:.3f}")
        
        self.logger.info(f"\nResults saved to: {self.args.result_dir}")


def main():
    """Main function to run SuperGPQA evaluation"""
    from config.argument_parser import config
    
    # Get arguments from centralized parser
    args = config()
    
    # Validate that we're running SuperGPQA evaluation
    if args.evaluation_type != "supergpqa":
        print(f"Warning: evaluation_type is set to '{args.evaluation_type}', but this is the SuperGPQA test runner")
        print("Setting evaluation_type to 'supergpqa'")
        args.evaluation_type = "supergpqa"
    
    # Initialize components (you'll need to set these up based on your environment)
    # agent = FunctionCallAgent(...)
    # evaluator = SuperGPQAEvaluator(...)
    
    # For now, just print the configuration
    print("SuperGPQA Evaluation Configuration:")
    print(f"Max samples: {args.max_samples}")
    print(f"Result dir: {args.result_dir}")
    print(f"Max steps: {args.max_steps}")
    
    # TODO: Initialize and run evaluation
    print("Please implement agent and evaluator initialization")


if __name__ == "__main__":
    main() 