from __future__ import annotations
import argparse
import datetime
import json
import logging
import os
import sys
import time
from multiprocessing import Process, Manager
from multiprocessing import current_process
from typing import List, Tuple

import lib_run_single
from desktop_env.desktop_env import DesktopEnv
from mm_agents.gui_agent_adapter import GUIAgentAdapter


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OSWorld with GUI-Agent adapter")

    # environment config
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument("--headless", action="store_true", help="Run in headless machine")
    parser.add_argument("--action_space", type=str, default="pyautogui", help="Action type")
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
        help="Observation type",
    )
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=15)

    # agent/model config
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_tokens", type=int, default=1500)
    parser.add_argument("--openai_api_key", type=str, default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--anthropic_api_key", type=str, default=os.getenv("ANTHROPIC_API_KEY", ""))

    # dataset config
    parser.add_argument("--test_config_base_dir", type=str, default="evaluation_examples")
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument("--test_all_meta_path", type=str, default="evaluation_examples/test_all.json")

    # logging / parallel
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--log_level", type=str, choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default='INFO')

    # provider config
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--provider_name", type=str, default="aws", choices=["aws","virtualbox","vmware","docker","azure"]) 
    parser.add_argument("--client_password", type=str, default="")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)

    return parser.parse_args()


def setup_logging(log_level: str) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    datetime_str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(os.path.join("logs", f"normal-{datetime_str}.log"), encoding="utf-8")
    debug_handler = logging.FileHandler(os.path.join("logs", f"debug-{datetime_str}.log"), encoding="utf-8")
    stdout_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.INFO)
    debug_handler.setLevel(logging.DEBUG)
    stdout_handler.setLevel(getattr(logging, log_level.upper()))
    formatter = logging.Formatter(fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s")
    for h in (file_handler, debug_handler, stdout_handler):
        h.setFormatter(formatter)
        logger.addHandler(h)
    return logging.getLogger("desktopenv.experiment")


def distribute_tasks(test_all_meta: dict) -> List[Tuple[str, str]]:
    tasks: List[Tuple[str, str]] = []
    for domain, examples in test_all_meta.items():
        for example_id in examples:
            tasks.append((domain, example_id))
    return tasks


def run_env_tasks(task_queue, args: argparse.Namespace, shared_scores):
    env = None
    try:
        from desktop_env.providers.aws.manager import IMAGE_ID_MAP
        REGION = args.region
        screen_size = (args.screen_width, args.screen_height)
        ami_id = IMAGE_ID_MAP[REGION].get(screen_size, IMAGE_ID_MAP[REGION][(1920, 1080)])
        env = DesktopEnv(
            path_to_vm=args.path_to_vm,
            action_space=args.action_space,
            provider_name=args.provider_name,
            region=REGION,
            snapshot_name=ami_id,
            screen_size=screen_size,
            headless=args.headless,
            os_type="Ubuntu",
            require_a11y_tree=args.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
            enable_proxy=True,
            client_password=args.client_password,
        )
        agent = GUIAgentAdapter(model=args.model, max_tokens=args.max_tokens, openai_api_key=args.openai_api_key or None, anthropic_api_key=args.anthropic_api_key or None)

        while True:
            try:
                item = task_queue.get(timeout=5)
            except Exception:
                break
            domain, example_id = item
            config_file = os.path.join(args.test_config_base_dir, f"examples/{domain}/{example_id}.json")
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    example = json.load(f)
            except Exception as e:
                logging.getLogger("desktopenv.experiment").error(f"Failed loading {config_file}: {e}")
                continue

            example_result_dir = os.path.join(
                args.result_dir,
                args.action_space,
                args.observation_type,
                args.model,
                domain,
                example_id,
            )
            os.makedirs(example_result_dir, exist_ok=True)
            try:
                lib_run_single.run_single_example(
                    agent,
                    env,
                    example,
                    args.max_steps,
                    example["instruction"],
                    args,
                    example_result_dir,
                    shared_scores,
                )
            except Exception as e:
                import traceback
                logging.getLogger("desktopenv.experiment").error(f"Error {domain}/{example_id}: {e}\n{traceback.format_exc()}")
                try:
                    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
                except Exception:
                    pass
                with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                    f.write(json.dumps({"Error": f"{domain}/{example_id} - {e}"}) + "\n")
    finally:
        try:
            if env:
                env.close()
        except Exception:
            pass


def main():
    args = config()
    logger = setup_logging(args.log_level)
    logger.info("Starting GUI-Agent adapter runner")
    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)
    if args.domain != "all":
        # Narrow to a single domain
        test_all_meta = {args.domain: test_all_meta.get(args.domain, [])}

    tasks = distribute_tasks(test_all_meta)
    with Manager() as manager:
        shared_scores = manager.list()
        task_queue = manager.Queue()
        for t in tasks:
            task_queue.put(t)

        procs: List[Process] = []
        for i in range(args.num_envs):
            p = Process(target=run_env_tasks, args=(task_queue, args, shared_scores), name=f"EnvProcess-{i+1}")
            p.daemon = True
            p.start()
            procs.append(p)

        try:
            while True:
                alive = [p for p in procs if p.is_alive()]
                if task_queue.empty() and not alive:
                    break
                time.sleep(3)
        finally:
            for p in procs:
                if p.is_alive():
                    p.terminate()
            for p in procs:
                p.join()

        scores = list(shared_scores)
        logger.info(f"Average score: {sum(scores) / len(scores) if scores else 0}")


if __name__ == "__main__":
    main()
