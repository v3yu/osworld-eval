"""Test runner for the GUI Agent"""
import argparse
import json
import logging
import os
import pickle
import re
from pathlib import Path
import base64
import io
from typing import List

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
from MMInA_evaluation.evaluator import evaluator_router
from utils.early_stop import early_stop
from utils.help_functions import is_domain_type, save_scores_to_json
from utils.training_data_collector import get_collector
from agent.llm_config import load_tool_llm


class TestRunner:
    """Handles the main test execution loop"""
    
    def __init__(self, args: argparse.Namespace, agent):
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
            args=args,  # Pass args to the environment
        )
        
        # Initialize tracking variables
        self.scores = {}
        self.trajSOM = {}
        self.trajImages = {}
        self.trajActions = {}
        self.trajSuccess = {}
        self.metrics_dict = {}
        
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
    
    def run(self, config_file_list: list[str]):
        """Run the main test loop"""
        # Load seen tasks
        seen = self._load_seen_tasks()
        
        # Initialize counters
        all_pass = 0
        all_all = 0
        
        # Process each config file
        for config_file in config_file_list:
            self._process_config_file(config_file, seen, all_pass, all_all)
        
        # Save results
        self._save_results()
        
        # Close environment
        self.env.close()
    
    def _load_seen_tasks(self):
        """Load seen tasks from instruction JSONs"""
        seen = set()
        for json_file in self.args.instruction_jsons:
            folder = os.path.split(json_file)[0]
            if os.path.exists(os.path.join(folder, "seen.json")):
                with open(os.path.join(folder, "seen.json")) as seen_file:
                    seen_data = json.load(seen_file)
                seen.update(seen_data["seen"])
        return seen
    
    def _process_config_file(self, config_file: str, seen: set, all_pass: int, all_all: int):
        """Process a single config file"""
        sub_domain = config_file.replace('/home/wenyi/MMInA/','').split('/')[1]
        self.scores[sub_domain] = {}
        self.scores[sub_domain]['task_scores'] = []
        self.scores[sub_domain]['hop_scores'] = []
        
        action_list = []

        render_helper = RenderHelper(config_file, self.args.result_dir)

        # Get intent and task info
        with open(config_file) as f:
            _c = json.load(f)
            intent = _c["intent"]    
            intent = intent.replace('https://library.kiwix.org/iewer#wikipedia_en_all_maxi_2024-01/A/User%3AThe_other_Kiwix_guy/Landing','https://www.wikipedia.org/')
            intent = intent.replace("http://localhost:7770/", "http://ec2-3-20-72-231.us-east-2.compute.amazonaws.com:7770/")
            # intent = intent.replace("For actions 'book a hotel','book a car', 'book a flight','search on the Youtube', 'search on the twitter', 'search some events', 'Find food', 'Travel Guide', 'Exchange dollars': the action is finished just after click the search button! Attention: If you think all the actions had been done, return the final url as the answer!!! \n\n Here are some reference urls: Wiki: https://library.kiwix.org/viewer#wikipedia_en_all_maxi_2024-01/A/User%3AThe_other_Kiwix_guy/Landing  \nRent a car:https://sg.trip.com/carhire/?channelid=14409&locale=en-SG&curr=USD \nBook a flight:https://www.momondo.com/ \nBook a hotel:https://sg.trip.com/hotels/?locale=en-SG&curr=USD \nShopping:http://localhost:7770/ \nSearch an event:https://www.eventbrite.com/ \nTwitter:https://twitter.com/home \nYoutube:https://www.youtube.com/ \nFind food:https://www.timeout.com/ \nExchange dollars: https://www.xe.com/ \nTravel Guide:https://www.nomadicmatt.com \n\n\n", '"')
            # if 'twitter' in intent:
            #     return
            if 'shopping' in intent:
                intent += "\n\n***NOTE: Please 1. directly search the product name 2.find the query product and click it 3. carefully check the product image and description to answer the question***"
            if 'compare' in config_file or 'normal' in config_file or 'multi567' in config_file or 'multipro' in config_file:
                intent = intent.replace('the action is finished just after click the search button!',
                                        'the action is finished when you successfully search the tarket hotel/ticket/flight/... with correct city, number of people, dates, etc.')
                intent += """***NOTE***: 
                1. Please always jump to multiple websites using the goto_url tool to complete the task. 
                2. Make sure you finish all the tasks. If you need to book a hotel/ticket/flight/..., you need to search the city, number of people, dates, etc.
                3. The task is finished until you see the final target, if you need to book a hotel/ticket/flight/...,  only when you see the destination hotel/ticket/flight/... with correct number of people, dates, etc. you can think the task is finished.
                4. Attention: If you think all the actions had been done, return the final url as the answer!!!
                """
            task_id = _c["task_id"]
            site = _c["sites"][0]
        
        episode_id = f"{site}_{task_id}"

        numbers = re.findall(r'\d+', config_file)
        self.args.task_cnt = int(numbers[0]) if numbers else None
        self.args.hop_cnt = 0
        
        self.logger.info(f"[Config file]: {config_file}")
        self.logger.info(f"[Intent]: {intent}")
        
        self.agent.reset(config_file)
        trajectory: Trajectory = []
        
        # Environment reset
        obs, info = self.env.reset(
            options={"config_file": config_file}, 
        )
        current_url = info["page"].url
        state_info: StateInfo = {"observation": obs, "info": info, "current_url": current_url}
        trajectory.append(state_info)
        print("CURRENT: ", current_url)
        
        meta_data = {"action_history": [],
                     "response_history": []}
        
        print("config_file: ", config_file)
        check_list = []
        
        cnt_ans = 0
        
        # Determine domain type and setup
        if is_domain_type(sub_domain, '2hop'):
            cnt_cl = 2
            with open(config_file, "r") as file:
                data = json.load(file)
                reference = data['eval']['reference_answers']['must_include']
            nxt = reference[0]
        elif is_domain_type(sub_domain, 'multihop'):
            with open(config_file, "r") as file:
                data = json.load(file)
                check_list = data['procedure']
                city = data['city']
                flight = data['flight']
                shop = data['shop']
            cnt_cl = len(check_list)
        elif is_domain_type(sub_domain, 'singlehop'):
            cnt_cl = 1

        # Calculate sum of total hop numbers of all tasks
        all_all += cnt_cl
        
        flag = True
        all_view_url = []
        count = 0
        
        # Information accumulation storage
        sub_query_answers = []
        
        # Start conversation for this task if training data collection is enabled
        if hasattr(self.agent, 'training_collector') and self.agent.training_collector:
            from utils.training_data_collector import get_collector
            collector = get_collector()
            if collector and collector.enabled:
                # Create conversation ID from task info
                conversation_id = f"{sub_domain}_{config_file.split('/')[-1].split('.')[0]}"
                collector.start_conversation(
                    conversation_id=conversation_id,
                    task_description=intent
                )
                self.logger.info(f"Started conversation collection for task: {conversation_id}")
        
        if self.args.subtask:
            # Implement subtask decomposition with LLM
            intent_list = self._decompose_task_into_subtasks(intent)
            self.logger.info(f"Task decomposed into {len(intent_list)} subtasks")
        else:
            intent_list = [intent]

        # Process each sub-query sequentially
        for sub_query_idx, current_intent in enumerate(intent_list):
            
            # Enhance current intent with previous subtask results if available
            if self.args.subtask and sub_query_idx > 0 and sub_query_answers:
                enhanced_intent = self._enhance_intent_with_previous_results(
                    current_intent, sub_query_answers, sub_query_idx
                )
                self.logger.info(f"[Subtask {sub_query_idx + 1}] Enhanced intent with previous results")
            else:
                enhanced_intent = current_intent
            
            # Reset environment for each sub-query if not the first one
            if sub_query_idx > 0:
                self.logger.info(f"[Sub-query {sub_query_idx + 1}] Resetting environment for new sub-query")
                obs, info = self.env.reset(options={"config_file": config_file})
                current_url = info["page"].url
                state_info: StateInfo = {"observation": obs, "info": info, "current_url": current_url}
                # Clear trajectory and start fresh for new sub-query
                trajectory = [state_info]
                meta_data = {"action_history": [],
                             "page": self.env.page}
                print("CURRENT: ", current_url)
                if not self.args.manual_action:
                    initial_plan = self.planner.generate_initial_plan(
                        query=enhanced_intent,
                        start_url=current_url
                    )
                    meta_data["initial_plan"] = initial_plan
            
            # Process current sub-query
            while True:
                current_url = current_url.lower()
                all_view_url.append(current_url)

                early_stop_flag, stop_info = early_stop(
                    trajectory, self.args.max_steps, {
                        "parsing_failure": self.args.parsing_failure_th,
                        "repeating_action": self.args.repeating_action_failure_th,
                    }
                )

                if early_stop_flag:
                    # Check if fallback is enabled and should be used
                    if self.args.enable_fallback:
                        from utils.fallback_answer import generate_fallback_answer                       
                        self.logger.info(f"[Fallback] Early stop detected: {stop_info}. Generating fallback answer...")
                        
                        # Generate fallback answer using the LLM
                        fallback_result = generate_fallback_answer(
                            question=enhanced_intent,
                            trajectory=trajectory,
                            model=self.agent.llm,
                            num_screenshots=self.args.fallback_screenshots
                        )
                        # Create a stop action with the fallback answer
                        action = create_stop_action(fallback_result['answer'])
                        # Integrate the fallback answer into the trajectory
                        trajectory.append(action)
                        sub_query_answers.append((enhanced_intent, fallback_result['answer']))
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
                        action, meta_data = action_self_check(gen_action, enhanced_intent, self.env.page, trajectory, max_retries=3, repeat_threshold=self.args.repeating_action_failure_th)
                    else:
                        action, meta_data = gen_action(enhanced_intent, meta_data)
                        
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
                
                # Draw action and intent on the image for debug or memory saving
                from PIL import ImageDraw, ImageFont, Image

                image_bytes = base64.b64decode(obs["image"])
                rendered_im = Image.open(io.BytesIO(image_bytes))
                draw = ImageDraw.Draw(rendered_im)
                # Example fonts; adjust if needed
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()

                draw.text((40, 40), action_str, fill=(0, 0, 0), font=font_large)
                draw.text((40, 80), current_intent, fill=(0, 0, 0), font=font_small)
                # Save the rendered image
                model_res_path = os.path.join(self.args.result_dir, self.args.model, self.args.domain)
                if self.args.hist:
                    model_res_path = os.path.join(model_res_path, f'hist_{self.args.hist_num}')
                task_res_path = os.path.join(model_res_path, f"task_{self.args.task_cnt}")
                hop_res_path = os.path.join(task_res_path, f"hop_{max(self.args.hop_cnt-1, 0)}")
                image_dir = os.path.join(hop_res_path, "images")
                # os.makedirs(image_dir, exist_ok=True)
                # rendered_im.save(os.path.join(image_dir, f"{count}.png"))

                if isinstance(action, list):
                    last_action_type = action[-1]["action_type"]
                else:
                    last_action_type = action["action_type"]
                if last_action_type in [ActionTypes.STOP, 'finished']:
                    self.logger.info(f"[Sub-query {sub_query_idx + 1}] Completed")
                    
                    # Store the subtask answer if using subtask decomposition
                    if self.args.subtask:
                        # Extract the answer from the stop action
                        if isinstance(action, list):
                            answer = action[-1].get('answer', '')
                        else:
                            answer = action.get('answer', '')
                        
                        # Store the subtask intent and answer
                        sub_query_answers.append((enhanced_intent, answer))
                        self.logger.info(f"[Subtask {sub_query_idx + 1}] Answer stored: {answer[:100]}...")
                    
                    break
                
                obs, _, terminated, _, info, current_url = self.env.step(action, observation=obs)
                # observation, 0.0, done, truncated, info
                print("CURRENT: ", current_url)

                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                if terminated:
                    # add a action place holder
                    trajectory.append(create_stop_action(""))
                    self.logger.info(f"[Sub-query {sub_query_idx + 1}] Terminated")
                    
                    # Store the subtask answer if using subtask decomposition
                    if self.args.subtask:
                        sub_query_answers.append((enhanced_intent, "Task terminated without completion"))
                        self.logger.info(f"[Subtask {sub_query_idx + 1}] Terminated without completion")
                    
                    break
                
                count += 1

        all_pass += cnt_ans

        # Store trajectory info
        self.trajActions[config_file] = action_list
        seen_episode = episode_id in seen

        # evaluate the scores
        evaluate_model = load_tool_llm(self.args)
        evaluator = evaluator_router(config_file, evaluate_model)
        score = evaluator(
            trajectory=trajectory,
            config_file=config_file,
            page=self.env.page,
            client=self.env.get_page_client(self.env.page),
        )
        last_action = trajectory[-1]
        pred = last_action.get("answer", "")
        reasoning = last_action.get("reasoning", "")
        self.logger.info(f"[Result] Predicted answer: {pred}\nReasoning: {reasoning}")
        
        self.metrics_dict[config_file] = {
                "config": config_file,
                "success": score,
                "seen": seen_episode,
            }
        self.trajSuccess[config_file] = score
        
        # Define success conditions for each domain type
        success_conditions = {
            'singlehop': lambda: score == 1,
            '2hop': lambda: cnt_ans == 2,
            'multihop': lambda: check_list[cnt_ans] == "end"
        }

        # Special handling for singlehop domain
        if is_domain_type(sub_domain, 'singlehop') and score == 1:
            cnt_ans += 1

        # Evaluate success based on domain type
        for domain_type, check_success in success_conditions.items():
            if is_domain_type(sub_domain, domain_type):
                passed = check_success()
                result = "PASS" if passed else "FAIL"
                self.logger.info(f"[Result] ({result}) {config_file}")
                self.scores[sub_domain]['task_scores'].append(int(passed))
                break
        else:
            raise NotImplementedError(f"Unsupported domain type: {sub_domain}")
        
        self.scores[sub_domain]['task_success_rate'] = sum(self.scores[sub_domain]['task_scores']) / len(self.scores[sub_domain]['task_scores'])
        # update the success rate for each subdomain
        self.logger.info(f"[Result] Current task success rate: {self.scores[sub_domain]['task_success_rate']}")
        
        if not is_domain_type(sub_domain, 'singlehop'):
            self.scores[sub_domain]['hop_scores'].append(f"{cnt_ans}/{cnt_cl}")
            self.logger.info(f"[Result] Current hop success rate: {all_pass}/{all_all}")
            self.scores[sub_domain]['hop_success_rate'] = f"{all_pass}/{all_all}"
        
        # if self.args.save_trace_enabled:
        #     self.env.save_trace(
        #         Path(self.args.result_dir) / "traces" / f"{task_id}.zip"
        #     )
        
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
                    "site": site,
                    "sub_domain": sub_domain,
                    "success": score,
                    "final_url": current_url,
                    "task_completed": True,
                    "task_description": intent
                }
                
                # End the conversation
                saved_file = collector.end_conversation(conversation_summary)
                if saved_file:
                    self.logger.info(f"Conversation saved: {saved_file}")
        
    def _save_results(self):
        """Save final results"""
        with open(Path(self.args.result_dir) / "scores.json", "w") as f:
            json.dump(self.scores, f, indent=4)
        
        if self.args.domain == 'full':
            save_scores_to_json(self.scores, self.args.result_dir)
    
    def _decompose_task_into_subtasks(self, task: str) -> List[str]:
        """Decompose a complex task into logical subtasks using LLM"""
        try:
            # Create a prompt for task decomposition
            system_prompt = """You are an expert at breaking down complex tasks into logical subtasks. 
            
IMPORTANT: 
1. Decompose the task into logical subtasks, NOT individual steps. Each subtask should be a meaningful unit of work that can be completed independently or with dependencies on previous subtasks.
2. Make sure the subtasks are not too granular, only return up to **3 subtasks**!
3. If the task is not complex, just return the original task as a single subtask.

For example:
- Task: "Whether Sydney Opera House is by the sea? if so book a hotel close to it"
- Good decomposition:
  1. "Check whether Sydney Opera House is by the sea"
  2. "Given the result of subtask 1, determine whether to book a hotel or stop"
  3. "If booking is needed, go to hotel booking website and book a hotel close to Sydney Opera House"

- Bad decomposition (too granular):
  1. "Open Wikipedia"
  2. "Search for Sydney Opera House"
  3. "Check location carefully"
  4. "Click on booking website"
  5. "Fill in hotel details"

Each subtask should be:
- Self-contained and meaningful
- Focused on a specific goal or decision point
- Written as a clear instruction that can be executed
- Dependent on previous subtask results when necessary

CRITICAL: Return subtasks separated by semicolons (;). Do not use line breaks or numbering.
Example format: "Check whether Sydney Opera House is by the sea; Given the result, determine whether to book a hotel; If booking needed, go to hotel website and book" """
            
            user_prompt = f"Decompose this task into logical subtasks:\n\n{task}"
            
            # Use the agent's LLM to decompose the task
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Access the LLM through the agent's configuration
            if hasattr(self.agent, 'llm') and self.agent.llm:
                response = self.agent.llm.chat(messages=messages, stream=False)
                decomposition = ""
                for resp in response:
                    if hasattr(resp, 'content'):
                        decomposition += resp.content
                    elif isinstance(resp, dict) and 'content' in resp:
                        # Handle dictionary response format
                        decomposition += resp['content']
                    else:
                        decomposition += str(resp)
                
                # # Collect training data
                # collector = get_collector()
                # if collector:
                #     collector.collect_interaction(
                #         messages=messages,
                #         response=response,
                #         context={"original_task": task}
                #     )
            else:
                # Fallback: use a simple decomposition
                self.logger.warning("LLM not available for task decomposition, using fallback")
                decomposition = f"{task}"
            
            # Parse the decomposition into a list using semicolon separation
            subtasks = []
            # Split by semicolon and clean up each subtask
            raw_subtasks = decomposition.strip().split(';')
            for subtask in raw_subtasks:
                subtask = subtask.strip().strip('"').strip("'")
                
                # Only add non-empty subtasks
                if subtask and len(subtask.strip()) > 0:
                    subtasks.append(subtask.strip())
            
            if not subtasks:
                # Fallback: return the original task as a single subtask
                self.logger.warning("Failed to decompose task, using original task as single subtask")
                return [task]
            
            return subtasks
            
        except Exception as e:
            self.logger.error(f"Error in task decomposition: {e}")
            # Fallback: return the original task as a single subtask
            return [task]
    
    def _enhance_intent_with_previous_results(self, current_intent: str, sub_query_answers: List[tuple], sub_query_idx: int) -> str:
        """Enhance the current intent with results from previous subtasks"""
        enhanced_intent = f"Current subtask ({sub_query_idx + 1}): {current_intent}\n\n"
        
        if sub_query_answers:
            enhanced_intent += "Previous subtask results:\n"
            for i, (subtask_intent, subtask_answer) in enumerate(sub_query_answers):
                enhanced_intent += f"Subtask {i + 1}: {subtask_intent}\n"
                enhanced_intent += f"Result: {subtask_answer}\n\n"
            
            enhanced_intent += "IMPORTANT: Use the above results from previous subtasks to inform your current action. "
            enhanced_intent += "If the current subtask depends on previous results, make sure to consider them in your decision-making.\n\n"
        
        enhanced_intent += f"Current task: {current_intent}"
        
        return enhanced_intent 