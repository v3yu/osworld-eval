"""Function Call Agent for GUI Agent using direct model calls with ReAct paradigm"""
import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from browser_env import Trajectory, Action
from browser_env.actions import ActionTypes
from actions import (
    create_click_action,
    create_type_action,
    create_scroll_action,
    create_wait_action,
    create_stop_action,
    create_none_action,
    create_key_press_action,
    create_goto_url_action,
    parse_action_json,
    # validate_action
)
from .llm_config import create_direct_model, load_tool_llm
from tools.gui_tools import ClickTool, TypeTool, ScrollTool, WaitTool, StopTool, PressKeyTool, PageGotoTool
from tools.analysis_tools import MapSearchTool, ContentAnalyzerTool
from tools.web_search_tools import WebSearchTool
from utils.training_data_collector import get_collector, set_collector
from utils.llm_wrapper import wrap_llm
from .manual_action import get_manual_action



class FunctionCallAgent:
    """Custom function call agent for GUI interactions using ReAct paradigm with direct model calls"""
    
    def __init__(self, args: argparse.Namespace, **kwargs):
        """Initialize the function call agent"""
        # Initialize direct model
        self.llm = create_direct_model(args)
        # Initialize tool LLM for tools that need it
        self.tool_llm = load_tool_llm(args)  
        
        # Define functions for the agent
        function_list = self._define_functions()
        # Build dynamic tool specs for prompt
        self.tool_specs = self._build_tool_specs(function_list)
        
        # Get system message
        self.system_message = self._get_system_message()
        
        self.args = args
        self.logger = logging.getLogger("logger")
        
        # Initialize function map for tools
        self.function_map = {}
        self._initialize_function_map()
        # if args.model == 'gpt-4o':
        #     print('*'*50, 'pop content_analyzer', '*'*50)
        #     self.function_map.pop('content_analyzer')
        print('*'*50, 'function_map', '*'*50)
        print(self.function_map)
        print('*'*50, 'function_map', '*'*50)
        
        # Initialize training data collector if enabled
        if hasattr(args, 'collect_training_data') and args.collect_training_data:
            from utils.training_data_collector import TrainingDataCollector
            training_data_dir = getattr(args, 'training_data_dir', 'training_data')
            self.training_collector = TrainingDataCollector(
                output_dir=training_data_dir,
                enabled=True
            )
            set_collector(self.training_collector)
            wrapped_llm = wrap_llm(self.llm)
            self.llm = wrapped_llm
        else:
            self.training_collector = None
        
        self.current_step = 0
        self.current_task = ""
        
        # Store analysis results and map search context for next steps
        self.last_analysis_result = None
        self.last_map_search_query = None
        self.last_map_search_result = None
        self.last_page_goto_name = None
        self.last_page_goto_result = None
        self.last_web_search_result = None
        self.last_web_search_screenshots = None

    
    def _define_functions(self) -> List[str]:
        """Define the functions available to the agent"""
        # Use function names instead of tool instances
        functions = [
            'click',
            'type', 
            'press_key',
            'scroll',
            'wait',
            'stop',
            'map_search',
            'content_analyzer',
            # 'goto_homepage',
            'goto_url',
            # 'google_web_search'
        ]
        return functions
    
    def _build_tool_specs(self, function_list: List[str]) -> List[Dict[str, Any]]:
        """Build tool specs (name, description, parameters) for prompt from function_list."""
        name_to_cls = {
            # Action tools
            'click': ClickTool,
            'type': TypeTool,
            'scroll': ScrollTool,
            'wait': WaitTool,
            'stop': StopTool,
            'press_key': PressKeyTool,
            # Analysis tools
            'map_search': MapSearchTool,
            'content_analyzer': ContentAnalyzerTool,
            # 'goto_homepage': GotoHomepageTool,
            'goto_url': PageGotoTool,
            # 'google_web_search': WebSearchTool
            }
            
        specs: List[Dict[str, Any]] = []
        for item in function_list:
            cls = name_to_cls.get(str(item))
            tool = None
            if cls is not None:
                # try:
                tool = cls()
                # except Exception:
                #     tool = None
            else:
                print(f"Tool {item} not found")
            if tool is not None:
                parameters = getattr(tool, 'parameters', None)
                if parameters is not None:
                    params_info = {}
                    parameters = parameters.get('properties', {})
                    for param_name, param_info in parameters.items():
                        params_info[param_name] = param_info
                    parameters['properties'] = params_info
                specs.append({
                    'name': getattr(tool, 'name', str(item)),
                    'description': getattr(tool, 'description', 'No description'),
                    'parameters': parameters
                })
    
        return specs
    
    def _initialize_function_map(self):
        """Initialize the function map with tool instances"""
        name_to_cls = {
            # Action tools
            'click': ClickTool,
            'type': TypeTool,
            'scroll': ScrollTool,
            'wait': WaitTool,
            'stop': StopTool,
            'press_key': PressKeyTool,
            # Analysis tools
            'map_search': MapSearchTool,
            'content_analyzer': ContentAnalyzerTool,
            # 'goto_homepage': GotoHomepageTool,
            'goto_url': PageGotoTool,
            # 'google_web_search': WebSearchTool
        }
        
        for name, cls in name_to_cls.items():
            try:
                tool = cls()
                tool.llm = self.tool_llm  # Set the tool LLM
                self.function_map[name] = tool
            except Exception as e:
                print(f"Failed to initialize tool {name}: {e}")
    
    def _normalize_intent(self, intent: str) -> str:
        """Return a stable task identity by stripping transient error feedback suffixes.
        Example: "Task...\n[ERROR_FEEDBACK]: ..." -> "Task..."
        """
        if not isinstance(intent, str):
            return ""
        marker = "\n[ERROR_FEEDBACK]:"
        idx = intent.find(marker)
        if idx != -1:
            intent = intent[:idx]
        return intent.strip()
    
    def _get_system_message(self) -> str:
        """Get the system message for the agent using ReAct paradigm"""
        tools_section = ""
        lines = []
        for spec in self.tool_specs:
            desc = spec.get('description') or ''
            name = spec.get('name')
            params = spec.get('parameters', {})
            
            # Build parameter description
            param_desc = ""
            if params and 'properties' in params:
                param_list = []
                for param_name, param_info in params['properties'].items():
                    param_type = param_info.get('type', 'string')
                    param_desc_text = param_info.get('description', '')
                    if 'enum' in param_info:
                        enum_values = ', '.join(param_info['enum'])
                        param_list.append(f"`{param_name}` ({param_type}: {enum_values})")
                    else:
                        param_list.append(f"`{param_name}` ({param_type}): {param_desc_text}")
                param_desc = f" - Parameters: {', '.join(param_list)}"
            
            lines.append(f"- **{name}**: {desc}{param_desc}")
        tools_section = "\n".join(lines)
        
        return f"""You are a GUI automation agent that can interact with web pages and applications using the ReAct (Reasoning and Acting) paradigm.

IMPORTANT: You MUST output your actions in structured JSON format that can be parsed directly. Use the function calling mechanism to execute actions.

Your task is to:
1. Analyze the current state of the page (including numerical labels on web elements)
2. Think through what needs to be done (Reasoning)
3. Determine the appropriate action to take (Acting)
4. Output the action in structured JSON format that can be parsed directly.

WORKFLOW GUIDELINES:
- If your previous action is type, then you must click related pages or scroll pages to find the information you need.
- When you need to search for information: Directly type your search query, then click the search button.
- After clicking an element, if you need to interact with it further (like typing), do so immediately
- Don't repeat the same action multiple times - if something doesn't work, try a different approach
- ALWAYS use function calling to execute actions - do not describe actions in text
- If the current page has no results, you must adjust your search term, especially make your search term simper and try again.

ACTION GUIDELINES:
1) To input text, NO need to click textbox first, directly type content. After typing, the system automatically hits `ENTER` key. Sometimes you should click the search button to apply search filters. Try to use simple language when searching.
2) You must distinguish between textbox and search button, don't type content into the button! If no textbox is found, you may need to click the search button first before the textbox is displayed.
3) Execute only one action per iteration.
4) STRICTLY avoid repeating the same action if the webpage remains unchanged. You may have selected the wrong web element or numerical label. Continuous use of Wait is also NOT allowed.
5) When a complex task involves multiple questions or steps, select "stop" only at the very end, after addressing all of these questions (steps). Flexibly combine your own abilities with the information in the web page.
6) Make sure you finish all the tasks. If you need to book a hotel/ticket/flight/..., you need to search the city, number of people, dates, etc.
7) The task is finished until you see the final target, if you need to book a hotel/ticket/flight/...,  only when you see the destination hotel/ticket/flight/... with correct number of people, dates, etc. you can think the task is finished.

WEB BROWSING GUIDELINES:
1) Don't interact with useless web elements like Login, Sign-in, donation that appear in webpages. Pay attention to key web elements like search textbox and menu.
2) Visit video websites like YouTube is allowed BUT you can't play videos. Clicking to download PDF is allowed and will be analyzed.
3) Pay attention to the filter and sort functions on the page, which, combined with scroll, can help you solve conditions like 'highest', 'cheapest', 'lowest', 'earliest', etc. Try your best to find the answer that best fits the task.

EXAMPLE WORKFLOW:
- Question: "I want to find the information about the product 'iPhone 15'"
- Action: {{"name": "type", "arguments": {{"text": "iPhone 15", "field_description": "search input field", "reasoning": "I need to search for iPhone 15 information"}}}}
- Action: {{"name": "click", "arguments": {{"description": "search button or first iPhone 15 result", "reasoning": "I need to click the search button or first result"}}}}
- Action: {{"name": "scroll", "arguments": {{"direction": "down", "reasoning": "I need to scroll to find more information"}}}}
- Action: {{"name": "stop", "arguments": {{"answer": "Found iPhone 15 information: [details]", "reasoning": "I have found the information, task complete"}}}}

- Question: "Whether the Sydney Opera House is open for the public? If so, book me a hotel in Sydney."
- Action: {{"name": "type", "arguments": {{"text": "Sydney Opera House", "field_description": "search input field", "reasoning": "I need to search for Sydney Opera House information"}}}}
- Action: {{"name": "click", "arguments": {{"description": "first result about Sydney Opera House", "reasoning": "I need to click the first result to get details"}}}}
- Action: {{"name": "scroll", "arguments": {{"direction": "down", "reasoning": "I need to scroll to find opening hours information"}}}}
- Action: {{"name": "goto_url", "arguments": {{"page_name": "hotel booking", "reasoning": "If it's open, I need to go to hotel booking page"}}}}
- Action: {{"name": "type", "arguments": {{"text": "Sydney", "field_description": "destination search field", "reasoning": "I need to search for hotels in Sydney"}}}}
- Action: {{"name": "click", "arguments": {{"description": "first hotel result", "reasoning": "I need to click the first hotel to book it"}}}}
- Action: {{"name": "stop", "arguments": {{"answer": "Booked hotel in Sydney: [hotel details]", "reasoning": "Task completed successfully"}}}}

Available actions:
{tools_section}

CRITICAL REQUIREMENTS:
1. ALWAYS use function calling - never describe actions in plain text
2. Provide clear reasoning in the 'reasoning' parameter of each function call
3. Be specific in descriptions to identify the correct elements
4. Use proper JSON format for all function arguments
5. Only execute one action at a time
6. If an action fails, try a different approach or element
7. For click and type actions, use the number label of the item you want to interact with in the description
8. Use simple search term when searching for information
9. If the current page has no results, you must adjust your search term, especially make your search term simper and try again.

Remember: Your responses must be structured function calls that can be parsed directly by the system."""

    def next_action_custom(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: Dict[str, Any],
        model=None,
        args: argparse.Namespace = None,
        reflection_messages=None
    ):
        """Generate the next action using function calling with ReAct paradigm"""
        
        # Update current task and step using normalized task id
        normalized_intent = self._normalize_intent(intent)
        if self.current_task != normalized_intent:
            self.current_task = normalized_intent
            self.current_step = 0
        
        self.current_step += 1
        
        max_retries = 3
        
        for attempt in range(max_retries):
            # Prepare messages for the LLM
            messages, meta_data = self._prepare_messages(trajectory, intent, meta_data)
            
            # Check if manual action mode is enabled
            if self.args.manual_action:
                responses = get_manual_action(trajectory, intent, meta_data)
                if self.training_collector:
                    self.training_collector.add_conversation_round(
                        messages=messages,
                        response=responses,
                        actual_model_input=messages,
                        functions=self.tool_specs,
                        round_info=None
                    )
            else:
                # Call the LLM with function calling
                responses = self.llm.chat(messages=messages, stream=False)
                if not isinstance(responses, list):
                    responses = [responses]
                if self.training_collector:
                    self.training_collector.add_conversation_round(
                        messages=messages,
                        response=responses,
                        round_info=None
                    )
            print('*'*50, 'responses', '*'*50)
            print(responses)
            print('*'*50, 'responses', '*'*50)
            meta_data['response_history'].extend(responses)
                    
            # Extract page if available in meta_data or elsewhere; pass explicitly
            page_for_tools = meta_data.get('page')
                
            # Process the response
            action = self._process_response(responses, trajectory, page_for_tools, intent)
            print('*'*50, 'action', '*'*50)
            print(action)
            print('*'*50, 'action', '*'*50)
            # waiting = input()
        
            # Check if action is None or empty
            if action is None or action.get('action_type') == ActionTypes.NONE:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}: Action is None, retrying...")
                    continue
                else:
                    print(f"Attempt {attempt + 1}: Action is None after {max_retries} attempts, returning None action")
                    return (create_none_action(), meta_data)
            else:
                # Valid action found, return it
                return (action, meta_data)
        
        # This should never be reached, but just in case
        return (create_none_action(), meta_data)
    
    def _prepare_messages(self, trajectory: Trajectory, intent: str, meta_data: Dict[str, Any]) -> List[Dict]:
        """Prepare messages for the LLM with ReAct context"""
        messages = []
        
        # Add system message    
        messages.append({
            'role': 'system',
            'content': self._get_system_message()
        })
        
        # Add current intent with ReAct prompt
        current_task = intent
        
        # Add analysis results context if available
        if self.last_analysis_result:
            # try:
            analysis_summary = self.last_analysis_result
            messages.append({
                'role': 'user',
                'content': analysis_summary
            })
            
            # Clear the analysis result after using it
            self.last_analysis_result = None
                
            # except Exception as e:
            #     self.logger.warning(f"Error processing analysis result: {e}")
            #     self.last_analysis_result = None
        
        # Add web search results context if available
        if self.last_web_search_result:
            web_search_summary = self.last_web_search_result
            
            # Create content with text and screenshots if available
            if self.last_web_search_screenshots:
                
                content_items = [
                    {"type": "text", "text": f"**Web search results:** {web_search_summary}"}
                ]
                
                # Add screenshot images and collect paths for cleanup
                screenshot_files_to_delete = []
                for screenshot_path in self.last_web_search_screenshots:
                    if os.path.exists(screenshot_path):
                        # Convert image to base64
                        import base64
                        with open(screenshot_path, 'rb') as img_file:
                            img_data = img_file.read()
                            img_base64 = base64.b64encode(img_data).decode('utf-8')
                            content_items.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            })
                        screenshot_files_to_delete.append(screenshot_path)
                
                messages.append({
                    'role': 'user',
                    'content': content_items
                })
                
                # Delete screenshot files after adding them to messages
                for screenshot_path in screenshot_files_to_delete:
                    try:
                        os.remove(screenshot_path)
                    except Exception as e:
                        continue
            else:
                messages.append({
                    'role': 'user',
                    'content': f"**Web search results:** {web_search_summary}"
                })
            
            # Clear the web search result after using it
            self.last_web_search_result = None
            self.last_web_search_screenshots = None
        
        # Add page goto results context if available
        if self.last_page_goto_result:
            page_goto_summary = f"Successfully navigated to {self.last_page_goto_name} at {self.last_page_goto_result}"
            messages.append({
                'role': 'user',
                'content': f"**Page navigation results:** {page_goto_summary}"
            })
            
            # Clear the page goto result after using it
            self.last_page_goto_result = None
            self.last_page_goto_name = None
        
        # Add recent trajectory information
        if trajectory:
            recent_obs = trajectory[-1]
            if isinstance(recent_obs, dict) and 'observation' in recent_obs:
                obs = recent_obs['observation']
                if 'image' in obs:
                    # Generate a description of the current page using LLM
                    page_description = self._generate_page_description(obs["image"])
                    
                    # Add the current screenshot with generated description
                    messages.append({
                        'role': 'user',
                        'content': [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{obs['image']}"
                                }
                            },
                            {"type": "text", "text": page_description}
                        ]
                    })
        # Add initial plan to the messages if available
        if meta_data and 'initial_plan' in meta_data:
            initial_plan = meta_data['initial_plan']
            messages.append({
                'role': 'user',
                'content': f"\n**Initial plan:**\n{initial_plan}\n Please contunue to the next step only if you have finished current step."
            })
        # # Add step plan to the messages
        # if self.planner and meta_data and 'initial_plan' in meta_data:
        #     step_plan = self.planner.step_plan(
        #         initial_plan=meta_data['initial_plan'],
        #         current_url=meta_data['url'] if hasattr(meta_data, 'url') else None,
        #         screenshot_base64=obs["image"],
        #         previous_actions=meta_data['action_history'][-5:],
        #         tool_specs=self.tool_specs
        #     )
        #     meta_data['step_plan'] = step_plan
        #     print("******************STEP PLAN******************")
        #     print(step_plan)
        #     print("******************STEP PLAN******************")
        #     messages.append({
        #         'role': 'user',
        #         'content': f"\n**Step Suggestions:**\n{step_plan}"
        #     })
        # Add current task to the messages
        
        # Add action history
        if meta_data and 'action_history' in meta_data:
            action_history = meta_data['action_history'][-5:]  # Last 5 actions
            response_history = meta_data['response_history'][-5:]
            history_text = ""
            if action_history:
                history_text += "**Recent Actions:**\n" + "\n".join(
                    f"{i+1}. {action}" for i, action in enumerate(action_history))
            if response_history:
                history_text += "\n\n**Recent Resoning Process:**\n" + "\n".join(
                    f"{i+1}. {response}" for i, response in enumerate(response_history))
            
            if history_text:
                history_text += "\n\n **Please carefully check the LOGIC of your previous actions and reasoning process, and what you have done! DO NOT REPEAT previous action sequences! If you observe repeat in your previous actions, DO ADJUST your action choice at this time!**"
                messages.append({
                    'role': 'user',
                    'content': history_text
                })
                
        messages.append({
            'role': 'user',
            'content': f"""Current task: {current_task}

IMPORTANT REMINDERS:
- Please specify the number label of the item you want to interact with, in the description of the action.
- Don't repeat the same action multiple times - try different approaches if something doesn't work
- If your previous action is type, then you must click related pages or scroll pages to find the information you need.
- If the current search term yields no results, you must adjust your search term and try again.
- Only output one action at a time, do not output multiple actions at once
- DO NOT USE the 'content_analyzer' tool! it is not working!!!
What would you like to do next?"""
                })
        for msg in messages:
            print('*'*50, msg['role'], '*'*50)
            if isinstance(msg['content'], list):
                for item in msg['content']:
                    if 'image_url' not in item:
                        print(item)
            else:
                print(msg['content'])
        
        return messages, meta_data
    
    
    def _process_response(self, responses: List[Dict], trajectory: Trajectory, page: Optional[Any] = None, intent: Optional[str] = None) -> Action:
        """Process the LLM response and convert to Action"""
            
        for response in responses:
            response = parse_action_json(response.content)
            print('*'*50, 'original parsed response', '*'*50)
            print(response)
            print('*'*50, 'original parsed response', '*'*50)
            if 'function_call' in response and response['function_call']:
                func_name = response['function_call']['name']
                func_args = response['function_call']['arguments']
                
                # Handle different function calls using the actions module
                if func_name == 'click':
                    return create_click_action(
                        description=func_args.get('description', ''),
                        reasoning=func_args.get('reasoning', '')
                    )
                elif func_name == 'type':
                    return create_type_action(
                        text=func_args.get('text', ''),
                        field_description=func_args.get('field_description', ''),
                        reasoning=func_args.get('reasoning', '')
                    )
                elif func_name == 'scroll':
                    return create_scroll_action(
                        direction=func_args.get('direction', 'down'),
                        reasoning=func_args.get('reasoning', '')
                    )
                elif func_name == 'wait':
                    return create_wait_action(
                        seconds=2.0,  # Default as requested
                        reasoning=func_args.get('reasoning', '')
                    )
                elif func_name == 'press_key':
                    return create_key_press_action(
                        key_comb=func_args.get('key', 'enter'),
                        reasoning=func_args.get('reasoning', '')
                    )
                elif func_name == 'stop':
                    answer = func_args.get('answer', 'Task completed')
                    self.logger.info(f"Agent answer: {answer}")
                    return create_stop_action(
                        answer=func_args.get('answer', 'Task completed'),
                        reasoning=func_args.get('reasoning', '')
                    )
                elif func_name == 'content_analyzer':
                    # Execute content analyzer and store results for next step
                    # Get the tool from function_map
                    tool = self.function_map.get(func_name)
                    if tool:
                        # Add trajectory context, page (if available), and LLM to kwargs
                        kwargs = {'page': page}
                        tool.llm = self.tool_llm
                        result = tool.call(json.dumps(func_args), **kwargs)
                        self.logger.info(f"Content analyzer result: {result}")
                        
                        # Store the analysis result for next step context
                        # ContentAnalyzerTool returns JSON string, so store it directly
                        self.last_analysis_result = result
                        
                        # Return a wait action to allow the agent to process the analysis result
                        return create_wait_action(
                            seconds=1.0,
                            reasoning=f"Content analysis completed. Analysis results will be available for the next step."
                        )
                    else:
                        self.logger.error(f"Tool {func_name} not found in function_map")
                        return create_none_action()
                    
                elif func_name == 'map_search':
                    # Execute map search tool which now returns a Google Maps URL
                    tool = self.function_map.get(func_name)
                    if tool:
                        result = tool.call(json.dumps(func_args))
                        # Expect result to be a URL string; try to extract
                        url = result.strip()
                        # Store context
                        self.last_map_search_query = func_args.get('query', '')
                        self.last_map_search_result = result
                        # If we got a URL, emit a goto action so the env updates the page
                        return create_goto_url_action(url)
                        
                elif func_name == 'goto_url':
                    # Execute page goto tool which returns a target URL
                    tool = self.function_map.get(func_name)
                    if tool:
                        tool.llm = self.tool_llm
                        result = tool.call(json.dumps(func_args))
                        # Expect result to be a URL string
                        url = result.strip()
                        # Store context
                        self.last_page_goto_name = func_args.get('page_name', '')
                        self.last_page_goto_result = result
                        # If we got a URL, emit a goto action so the env updates the page
                        return create_goto_url_action(url)
                        
                elif func_name == 'google_web_search':
                    # Execute web search tool and store results for next step
                    tool = self.function_map.get(func_name)
                    if tool:
                        # Set the LLM for the web search tool
                        tool.set_llm(self.tool_llm)
                        
                        # Handle async call
                        import asyncio
                        import concurrent.futures
                        
                        # Check if event loop is already running
                        try:
                            # Try to get running loop
                            asyncio.get_running_loop()
                            # Loop is running, use ThreadPoolExecutor
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(asyncio.run, tool.call(json.dumps(func_args)))
                                result = future.result(timeout=120)  # 2 minute timeout
                        except RuntimeError:
                            # No loop running, can use asyncio.run directly
                            result = asyncio.run(tool.call(json.dumps(func_args)))
                        self.logger.info(f"Web search result: {result}")
                        
                        # Extract screenshot information from result
                        screenshot_paths = []
                        if result and "[Screenshot available:" in result:
                            import re
                            screenshot_matches = re.findall(r'\[Screenshot available: ([^\]]+)\]', result)
                            screenshot_paths = screenshot_matches
                            self.logger.info(f"Found screenshots: {screenshot_paths}")
                        
                        # Store the web search result and screenshots for next step context
                        self.last_web_search_result = result
                        self.last_web_search_screenshots = screenshot_paths
                        
                        # Return a wait action to allow the agent to process the search result
                        return create_wait_action(
                            seconds=1.0,
                            reasoning=f"Web search completed. Search results will be available for the next step."
                        )
                        
            else:
                print('*'*50, 'no function call', '*'*50)
        # If no function call, try to parse natural language responses
        for response in responses:
            # if 'content' in response and response['content']:
            content = response.content
            print(f"Parsing natural language content: {content}")
            
            # # First try to match action patterns in the content
            # action = self._parse_natural_language_action(content)
            # if action and action.get('action_type') != ActionTypes.NONE:
            #     print(f"Successfully parsed natural language action: {action}")
            #     return action
            
            # If pattern matching fails, use LLM to parse the content
            action = self._parse_natural_language_with_llm(content, page)
            if action and action.get('action_type') != ActionTypes.NONE:
                print(f"Successfully parsed natural language action with LLM: {action}")
                return action
        
        # If no function call and no natural language action found, return none action
        action = create_none_action()
        

        
        return action
    

    def _parse_natural_language_with_llm(self, content: str, page: Optional[Any] = None) -> Action:
        """Use LLM to parse natural language content and extract action information"""
        try:
            # Create a prompt for the LLM to parse the content
            system_prompt = """You are an expert at parsing natural language responses and converting them into structured actions for a GUI automation agent.

Available actions:
- click: Click on elements by describing what you want to click
- type: Type text into input fields by describing the field
- goto_url: Navigate to a specific URL
- press_key: Press specific keys (enter, delete, space, etc.)
- scroll: Scroll the page in different directions (up, down, left, right)
- wait: Wait for a specified number of seconds
- stop: Stop the task and provide final answer
- map_search: Navigate to Google Maps for geographical searches
- content_analyzer: Analyze page content and images (results will be available for next step)

CRITICAL: You MUST respond with ONLY a valid JSON object. No other text, no explanations, no markdown formatting, just pure JSON.

Parse the given content and return a JSON object with the following structure:
{
    "action_type": "click|type|press_key|scroll|wait|stop|goto_url|map_search|content_analyzer",
    "description": "description of what to click",
    "text": "text to type (for type action)",
    "field_description": "description of the field (for type action)",
    "page_name": "name of the page (for goto_url action)",
    "key": "key to press (for press_key action)",
    "direction": "scroll direction (for scroll action)",
    "seconds": "number of seconds (for wait action)",
    "answer": "final answer (for stop action)",
    "query": "query to search (for map_search action)",
    "reasoning": "why this action is needed"
}

EXAMPLES:

For a click action:
{
    "action_type": "click",
    "description": "the search button",
    "reasoning": "Need to click the search button to submit the query"
}

For a type action:
{
    "action_type": "type",
    "text": "Sydney Opera House",
    "field_description": "the search input field",
    "reasoning": "Need to type the search query into the search field"
}

For a press_key action:
{
    "action_type": "press_key",
    "key": "enter",
    "reasoning": "Need to press the enter key to submit the query"
}


For a scroll action:
{
    "action_type": "scroll",
    "direction": "down",
    "reasoning": "Need to scroll down to load more content"
}

For a wait action:
{
    "action_type": "wait",
    "seconds": 2.0,
    "reasoning": "Need to wait for 2 seconds to load the page"
}


For a stop action (task completion):
{
    "action_type": "stop",
    "answer": "Yes, there is a Ferris wheel in the center of Shanghai. It is the Sky Ring Ferris Wheel in Joy City Shanghai.",
    "reasoning": "The information confirms the presence of the Sky Ring Ferris Wheel in the center of Shanghai"
}

For a map_search action:
{
    "action_type": "map_search",
    "query": "Sydney Opera House",
    "reasoning": "Need to search for the Sydney Opera House on Google Maps"
}

For a google_web_search action:
{
    "action_type": "google_web_search",
    "query": "Sydney Opera House",
    "reasoning": "Need to search for the Sydney Opera House on Google"
}

For a content_analyzer action:
{
    "action_type": "content_analyzer",
    "reasoning": "Need to analyze the page content and images"
}


IMPORTANT: If the content indicates that a task is complete or provides a final answer, use action_type "stop" with the answer in the "answer" field.

REMEMBER: Output ONLY valid JSON, nothing else."""

            user_prompt = f"Parse this content and extract the action:\n\n{content}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Call the LLM
            response = self.tool_llm.chat(messages=messages, stream=False)
            result = ""
            for resp in response:
                if hasattr(resp, 'content'):
                    result += resp.content
                elif isinstance(resp, dict) and 'content' in resp:
                    # Handle dictionary response format
                    result += resp['content']
                else:
                    result += str(resp)
            result = result.strip('```json').strip('```').replace("'", '"')
            print(f"LLM response: {result}")
            
            # Use regex to extract individual fields from LLM output
            import json
            import re
            
            # Extract individual fields using regex
            action_data = {}
            
            # Extract action_type
            action_type_match = re.search(r'"action_type":\s*"([^"]+)"', result, re.IGNORECASE)
            if action_type_match:
                action_data['action_type'] = action_type_match.group(1).lower()
            
            # Extract reasoning
            reasoning_match = re.search(r'"reasoning":\s*"([^"]+)"', result, re.IGNORECASE)
            if reasoning_match:
                action_data['reasoning'] = reasoning_match.group(1)
            
            # Extract query (for search actions)
            query_match = re.search(r'"query":\s*"([^"]+)"', result, re.IGNORECASE)
            if query_match:
                action_data['query'] = query_match.group(1)
            
            # Extract description (for click actions)
            description_match = re.search(r'"description":\s*"([^"]+)"', result, re.IGNORECASE)
            if description_match:
                action_data['description'] = description_match.group(1)
            
            # Extract text (for type actions)
            text_match = re.search(r'"text":\s*"([^"]+)"', result, re.IGNORECASE)
            if text_match:
                action_data['text'] = text_match.group(1)
            
            # Extract field_description (for type actions)
            field_desc_match = re.search(r'"field_description":\s*"([^"]+)"', result, re.IGNORECASE)
            if field_desc_match:
                action_data['field_description'] = field_desc_match.group(1)
                
            # Extract url (for goto_url actions)
            url_match = re.search(r'"url":\s*"([^"]+)"', result, re.IGNORECASE)
            if url_match:
                action_data['url'] = url_match.group(1)
            else:
                page_name_match = re.search(r'"page_name":\s*"([^"]+)"', result, re.IGNORECASE)
                if page_name_match:
                    page_name = page_name_match.group(1).lower()
                    page_urls = {
                        'wiki': 'https://www.wikipedia.org/',
                        'ticket': 'https://www.trip.com/flights/',
                        'car': 'https://sg.trip.com/carhire/?channelid=14409&locale=en-SG&curr=USD',
                        'flight': 'https://www.momondo.com/',
                        'hotel': 'https://sg.trip.com/hotels/?locale=en-SG&curr=USD',
                        'shopping': 'http://ec2-3-20-72-231.us-east-2.compute.amazonaws.com:7770//',
                        'event': 'https://www.eventbrite.com/',
                        'map': 'https://www.google.com/maps',
                        'youtube': 'https://www.youtube.com/',
                        'food': 'https://www.timeout.com/',
                        'travel': 'https://www.nomadicmatt.com/',
                        'dollars': 'https://www.xe.com/',
                        'twitter': 'https://twitter.com/home',
                    }
                    if 'car' in page_name:
                        page_name = 'car'
                    elif 'wiki' in page_name:
                        page_name = 'wiki'
                    elif 'ticket' in page_name:
                        page_name = 'ticket'
                    elif 'flight' in page_name:
                        page_name = 'flight'
                    elif 'hotel' in page_name:
                        page_name = 'hotel'
                    elif 'shopping' in page_name:
                        page_name = 'shopping'
                    elif 'event' in page_name:
                        page_name = 'event'
                    elif 'map' in page_name:
                        page_name = 'map'
                    elif 'youtube' in page_name:
                        page_name = 'youtube'
                    elif 'food' in page_name:
                        page_name = 'food'
                    elif 'travel' in page_name:
                        page_name = 'travel'
                    elif 'dollars' in page_name:
                        page_name = 'dollars'
                    elif 'twitter' in page_name:
                        page_name = 'twitter'
                    try:
                        action_data['url'] = page_urls[page_name]
                    except:
                        action_data['url'] = ''
            
            # Extract key (for press_key actions)
            key_match = re.search(r'"key":\s*"([^"]+)"', result, re.IGNORECASE)
            if key_match:
                action_data['key'] = key_match.group(1)
            
            # Extract direction (for scroll actions)
            direction_match = re.search(r'"direction":\s*"([^"]+)"', result, re.IGNORECASE)
            if direction_match:
                action_data['direction'] = direction_match.group(1)
            
            # Extract seconds (for wait actions)
            seconds_match = re.search(r'"seconds":\s*([0-9.]+)', result, re.IGNORECASE)
            if seconds_match:
                action_data['seconds'] = float(seconds_match.group(1))
            
            # Extract answer (for stop actions)
            answer_match = re.search(r'"answer":\s*"([^"]+)"', result, re.IGNORECASE)
            if answer_match:
                action_data['answer'] = answer_match.group(1)
            
            # Check if it's a null response
            if re.search(r'\bnull\b', result, re.IGNORECASE):
                return create_none_action()
            
            # Check if we found any action data
            if not action_data or 'action_type' not in action_data:
                print("No valid action data found in LLM response")
                return create_none_action()
            
            print(f"Extracted action data: {action_data}")
            if action_data is None:
                return create_none_action()
            
            # Convert to Action based on action_type
            action_type = action_data.get('action_type', '').lower()
            reasoning = action_data.get('reasoning', '')
            
            if action_type == 'click':
                return create_click_action(
                    description=action_data.get('description', ''),
                    reasoning=reasoning
                )
            elif action_type == 'type':
                return create_type_action(
                    text=action_data.get('text', ''),
                    field_description=action_data.get('field_description', ''),
                    reasoning=reasoning
                )
            elif action_type == 'press_key':
                return create_key_press_action(
                    key_comb=action_data.get('key', 'enter'),
                    reasoning=reasoning
                )
            elif action_type == 'scroll':
                return create_scroll_action(
                    direction=action_data.get('direction', 'down'),
                    reasoning=reasoning
                )
            elif action_type == 'wait':
                return create_wait_action(
                    seconds=float(action_data.get('seconds', 2.0)),
                    reasoning=reasoning
                )
            elif action_type == 'stop':
                return create_stop_action(
                    answer=action_data.get('answer', 'Task completed'),
                    reasoning=reasoning
                )
            elif action_type == 'map_search':
                tool = self.function_map.get(action_type)
                if tool:
                    func_args = {
                        'query': action_data.get('query', ''),
                        'reasoning': action_data.get('reasoning', '')
                    }
                    result = tool.call(json.dumps(func_args))
                    # Expect result to be a URL string; try to extract
                    url = result.strip()
                    # Store context
                    self.last_map_search_query = func_args.get('query', '')
                    self.last_map_search_result = result
                    # If we got a URL, emit a goto action so the env updates the page
                    return create_goto_url_action(url)
            elif action_type == 'goto_url':
                return create_goto_url_action(
                    url=action_data.get('url', '')
                )
            elif action_type == 'google_web_search':
                # Set the LLM for the web search tool
                    tool.set_llm(self.tool_llm)
                    # Handle async call
                    import asyncio
                    import concurrent.futures
                    
                    # Check if event loop is already running
                    try:
                        # Try to get running loop
                        asyncio.get_running_loop()
                        # Loop is running, use ThreadPoolExecutor
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(asyncio.run, tool.call(json.dumps(func_args)))
                            result = future.result(timeout=120)  # 2 minute timeout
                    except RuntimeError:
                        # No loop running, can use asyncio.run directly
                        result = asyncio.run(tool.call(json.dumps(func_args)))
                    self.logger.info(f"Web search result: {result}")
                    
                    # Extract screenshot information from result
                    screenshot_paths = []
                    if result and "[Screenshot available:" in result:
                        import re
                        screenshot_matches = re.findall(r'\[Screenshot available: ([^\]]+)\]', result)
                        screenshot_paths = screenshot_matches
                        self.logger.info(f"Found screenshots: {screenshot_paths}")
                    
                    # Store the web search result and screenshots for next step context
                    self.last_web_search_result = result
                    self.last_web_search_screenshots = screenshot_paths
                    
                    # Return a wait action to allow the agent to process the search result
                    return create_wait_action(
                        seconds=1.0,
                        reasoning=f"Web search completed. Search results will be available for the next step."
                    )
            elif action_type == 'content_analyzer':
                tool = self.function_map.get(action_type)
                func_args = {
                    'query': action_data.get('query', ''),
                    'reasoning': action_data.get('reasoning', '')
                }
                if tool:
                    # Add trajectory context, page (if available), and LLM to kwargs
                    kwargs = {'page': page}
                    tool.llm = self.tool_llm
                    result = tool.call(json.dumps(func_args), **kwargs)
                    self.logger.info(f"Content analyzer result: {result}")
                    
                    # Store the analysis result for next step context
                    # ContentAnalyzerTool returns JSON string, so store it directly
                    self.last_analysis_result = result
                    
                    # Return a wait action to allow the agent to process the analysis result
                    return create_wait_action(
                        seconds=1.0,
                        reasoning=f"Content analysis completed. Analysis results will be available for the next step."
                    )
            else:
                return create_none_action()
            
        except Exception as e:
            print(f"Error in LLM parsing: {e}")
            return create_none_action()

    def _generate_page_description(self, image_base64: str) -> str:
        """Generate a description of the current page using the LLM"""
        try:
            # Create a prompt for the LLM to describe the page
            messages = [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant that analyzes web page screenshots and provides clear, concise descriptions of what you see. Focus on the main content, interactive elements, and overall purpose of the page.'
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Please describe this web page screenshot. Include the main content, any visible buttons, forms, or interactive elements, and the overall purpose of the page."
                        }
                    ]
                }
            ]
            
            # Get LLM response
            response = self.tool_llm.chat(messages=messages, stream=False)
            if hasattr(response, 'content'):
                description = response.content
            else:
                description = str(response)
            
            description = description.replace("\"text\": \"{'role': 'assistant', 'content': '", "")
            description = description.replace("'}\"", "")
            description = description[:2000]
            return description if description else "Current page state - analyze this and decide what to do next"
            
        except Exception as e:
            self.logger.warning(f"Error generating page description: {e}")
            return "Current page state - analyze this and decide what to do next"
    
    def reset(self, test_config_file: str) -> None:
        """Reset the agent for a new task"""
        self.logger.info(f"Resetting agent for config file: {test_config_file}")
        # Clear any internal state if needed
        pass


def construct_agent(args: argparse.Namespace) -> FunctionCallAgent:
    """Construct a function call agent"""
    agent = FunctionCallAgent(args)
    return agent 