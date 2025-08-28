"""Main script to run the GUI Agent"""
import argparse
import logging
import time
from pathlib import Path

from agent import construct_agent
from browser_env import ScriptBrowserEnv, create_stop_action
from browser_env.actions import is_equivalent
from config.argument_parser import config
from MMInA_evaluation.evaluator import evaluator_router
from MMInA_evaluation.test_runner import TestRunner
from SuperGPQA_evaluation.test_runner import SuperGPQATestRunner
from WebWalkerQA_evaluation.test_runner import WebWalkerQATestRunner
from utils.early_stop import early_stop
from utils.help_functions import prepare, set_global_variables, MMINA_DICT, get_unfinished
from utils.logging_setup import setup_logging
from utils.action_check import action_self_check

import sys
sys.path.append('/lustre/scratch/users/guangyi.liu/agent/Memory_Web_Agent')

# Import the new grounding model loader
from agent.llm_config import load_grounding_model_vllm

def main():
    """Main execution function"""
    args = config()
    args.sleep_after_execution = 2.5
    
    # Setup logging
    datetime, LOG_FILE_NAME, logger = setup_logging(args)
    set_global_variables(datetime, LOG_FILE_NAME, logger)
    
    # Prepare environment
    prepare(args)
    logger.info(f"Observation context length: {args.max_obs_length}")
    logger.info(f"Multi-Modality: {args.multimodal}")
    
    # Load model and agent
    # model, tokenizer = load_model(args)
    # args.loaded_model = model
    # args.loaded_tokenizer = tokenizer
    
    # Load grounding model using vLLM
    grounding_model = load_grounding_model_vllm(args)
    args.grounding_model = grounding_model
    
    agent = construct_agent(args)
    
    # Run evaluation based on type
    if args.evaluation_type == "supergpqa":
        run_supergpqa_evaluation(args, agent, logger)
    elif args.evaluation_type == "webwalkerqa":
        run_webwalkerqa_evaluation(args, agent, logger)
    elif args.evaluation_type == "mmina":
        # Default MMInA evaluation
        test_file_list = prepare_test_files(args)
        test_file_list = test_file_list[:50]
        test_file_list = get_unfinished(test_file_list, args.result_dir)
        logger.info(f"Total {len(test_file_list)} tasks to process")
        run_tests(args, agent, test_file_list)
    
    logger.info(f"Test finished. Log file: {LOG_FILE_NAME}")


def prepare_test_files(args):
    """Prepare the list of test files to process"""
    test_file_list = []
    assert args.domain in ['full', 'shopping', 'wikipedia', 'normal', 'multi567', 'compare', 'multipro'], f"Invalid domain {args.domain}"
    task_num = MMINA_DICT[args.domain]
    
    if args.domain == 'full':
        for domain in ['shopping', 'wikipedia', 'normal', 'multi567', 'compare', 'multipro']:
            sub_task_num = MMINA_DICT[domain]
            for i in range(sub_task_num):
                test_file_list.append(f"mmina/{domain}/{i+1}.json")
    else:
        test_file_list = [f"mmina/{args.domain}/{i+1}.json" for i in range(task_num)]
    
    return test_file_list


def run_supergpqa_evaluation(args, agent, logger):
    """Run SuperGPQA evaluation"""
    logger.info("Starting SuperGPQA evaluation...")
    
    # Create result directory
    from pathlib import Path
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize and run SuperGPQA test runner
    test_runner = SuperGPQATestRunner(args, agent)
    test_runner.run()
    
    logger.info("SuperGPQA evaluation completed!")


def run_webwalkerqa_evaluation(args, agent, logger):
    """Run WebWalkerQA evaluation"""
    logger.info("Starting WebWalkerQA evaluation...")
    
    # Create result directory
    from pathlib import Path
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize and run WebWalkerQA test runner
    test_runner = WebWalkerQATestRunner(args, agent)
    test_runner.run()
    
    logger.info("WebWalkerQA evaluation completed!")


def run_tests(args, agent, config_file_list):
    """Run the main test loop"""
    
    test_runner = TestRunner(args, agent)
    test_runner.run(config_file_list)


if __name__ == "__main__":
    main()