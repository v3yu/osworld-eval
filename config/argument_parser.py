"""Argument parser configuration for the GUI Agent"""
import argparse


def config() -> argparse.Namespace:
    """Configure and parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    
    # Browser environment arguments
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--observation_type",
        choices=["accessibility_tree", "html", "image"],
        default="image",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=720)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument(
        "--imgbin_dir",
        type=str,
        default="",
    ) # Not in use

    # Agent configuration
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--parsing_failure_th",
        help="When consecutive parsing failure exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When consecutive repeating action exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument("--domain", type=str, default="shopping", 
                       choices=['full', 'normal', 'multi567', 'compare', 'multipro', 'shopping', 'wikipedia'])
    parser.add_argument("--hist", action='store_true', default=False)
    parser.add_argument("--hist_fold", type=str, default="./cache/history/")
    parser.add_argument("--hist_num", type=int, default=1)

    parser.add_argument("--task_cnt", type=int, default=0)
    parser.add_argument("--hop_cnt", type=int, default=0)
    
    # OpenAI API key
    parser.add_argument("--openai_api_key", type=str, default="")
    
    # Language model configuration
    parser.add_argument("--provider", type=str, default="custom")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--loaded_tokenizer", default=None)
    parser.add_argument("--loaded_model", default=None)
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=10000)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument("--multimodal", type=str, default='multimodal', 
                       help="Whether to use multimodal input", 
                       choices=['multimodal', 'text', 'continuous_memory'])
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=1920,
    )
    parser.add_argument("--add_history_num", type=int, default=5, 
                       help="Whether to add history actions to the prompt")
    parser.add_argument("--multimodal_history", type=bool, default=True, 
                       help="Whether to use multimodal history in the prompt")
    parser.add_argument("--save_examples_memory", action='store_true', default=False, 
                       help="Whether to add example memory to the agent")
    parser.add_argument("--instruction_jsons", type=str, nargs='+', default=[], 
                       help="jsons to use for example retrieval")
    
    # Example configuration
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=1000)

    # Logging related
    parser.add_argument("--result_dir", type=str, default="")
    
    parser.add_argument('--action_check', action='store_true', default=True, 
                       help='Enable action self-check and retry for model actions')
    
    # Fallback answer generation
    parser.add_argument("--enable_fallback", action='store_true', default=False, 
                       help="Enable fallback answer generation when agent is early stopped")
    parser.add_argument("--fallback_screenshots", type=int, default=5, 
                    help="Number of latest screenshots to use for fallback answer")
    
    # Training data collection configuration
    parser.add_argument("--collect_training_data", action='store_true', default=True,
                        help="Enable collection of training data (prompts and responses)")
    parser.add_argument("--training_data_dir", type=str, default="training_data",
                        help="Directory to save training data files")
    # Planning module configuration
    parser.add_argument("--self_plan", action='store_true', default=False, 
                       help="Enable self-planning module for dynamic plan generation and fact management")
    parser.add_argument("--planning_interval", type=int, default=5, 
                       help="Interval between planning steps (number of actions)")
    
    # Subtask decomposition configuration
    parser.add_argument("--subtask", action='store_true', default=False, 
                       help="Enable subtask decomposition for complex task breakdown")
    
    # Evaluation configuration
    parser.add_argument("--evaluation_type", type=str, default="webwalkerqa", 
                       choices=['mmina', 'supergpqa', 'webwalkerqa'],
                       help="Type of evaluation to run")
    parser.add_argument("--max_samples", type=int, default=100, 
                       help="Maximum number of samples to test (for SuperGPQA and WebWalkerQA)")
    parser.add_argument("--render_screenshot", action='store_true', 
                       help="Render screenshots during evaluation")
    
    # WebWalkerQA evaluation configuration
    parser.add_argument("--webwalkerqa_split", type=str, default="silver", 
                       choices=['main', 'silver'])
    
    args = parser.parse_args()
    args.instruction_path = 'agent/prompts/jsons/p_cot_ground_actree_2s.json'
    
    # Set result directory based on evaluation type
    if not args.result_dir:
        if args.evaluation_type == "supergpqa":
            plan_suffix = "_self_plan" if args.self_plan else ""
            subtask_suffix = "_subtask" if args.subtask else ""
            args.result_dir = f'results/supergpqa_{args.model}{plan_suffix}{subtask_suffix}'
        elif args.evaluation_type == "webwalkerqa":
            plan_suffix = "_self_plan" if args.self_plan else ""
            subtask_suffix = "_subtask" if args.subtask else ""
            args.result_dir = f'results/webwalkerqa_{args.webwalkerqa_split}_{args.model}{plan_suffix}{subtask_suffix}'
        elif args.evaluation_type == "mmina":
            plan_suffix = "_self_plan" if args.self_plan else ""
            subtask_suffix = "_subtask" if args.subtask else ""
            args.result_dir = f'results/{args.domain}_{args.model}{plan_suffix}{subtask_suffix}'
    
    # Set training data directory
    if args.evaluation_type == "supergpqa":
        args.training_data_dir = f"training_data/supergpqa"
    elif args.evaluation_type == "webwalkerqa":
        args.training_data_dir = f"training_data/webwalkerqa"
    elif args.evaluation_type == "mmina":
        args.training_data_dir = f"training_data/{args.domain}"
    
    return args 