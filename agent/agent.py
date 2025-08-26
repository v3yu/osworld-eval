"""Function Call Agent for GUI Agent using Qwen-Agent framework with ReAct paradigm"""
import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import sys 
sys.path.append('/lustre/scratch/users/guangyi.liu/agent/Qwen-Agent')
from qwen_agent import Agent
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import get_chat_model
from qwen_agent.llm.schema import Message, ContentItem
from qwen_agent.tools import BaseTool
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
    # validate_action
)
from .llm_config import configure_llm
from tools.gui_tools import ClickTool, TypeTool, ScrollTool, WaitTool, StopTool, PressKeyTool, PageGotoTool
from tools.analysis_tools import MapSearchTool, ContentAnalyzerTool
from tools.web_search_tools import WebSearchTool
from utils.training_data_collector import get_collector, set_collector
from utils.llm_wrapper import wrap_llm



class FunctionCallAgent(FnCallAgent):
    """Custom function call agent for GUI interactions using ReAct paradigm"""
    
    def __init__(self, args: argparse.Namespace, **kwargs):
        """Initialize the function call agent"""
        # Configure LLM for vLLM with qwen2.5-instruct-VL
        llm_config = self._configure_llm(args)
        
        # Define functions for the agent
        function_list = self._define_functions()
        
        # Build dynamic tool specs for prompt
        tool_specs = self._build_tool_specs(function_list)
        
        # Initialize parent class
        super().__init__(
            function_list=function_list,
            llm=llm_config,
            system_message=self._get_system_message(tool_specs),
            name="GUI_Function_Call_Agent",
            description="A function call agent for GUI interactions using ReAct paradigm",
            **kwargs
        )
        
        self.args = args
        self.logger = logging.getLogger("logger")
        
        # Initialize training data collector if enabled
        if hasattr(args, 'collect_training_data') and args.collect_training_data:
            from utils.training_data_collector import TrainingDataCollector
            training_data_dir = getattr(args, 'training_data_dir', 'training_data')
            self.training_collector = TrainingDataCollector(
                output_dir=training_data_dir,
                enabled=True
            )
            set_collector(self.training_collector)
        else:
            self.training_collector = None
        
        # Wrap LLM for training data collection if enabled
        if self.training_collector:
            wrapped_llm = wrap_llm(self.llm)
            self.llm = wrapped_llm
        
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

    def _configure_llm(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Configure LLM using the centralized configuration"""
        return configure_llm(args)
    
    def _define_functions(self) -> List[Union[str, BaseTool]]:
        """Define the functions available to the agent"""
        # Use custom tool instances instead of function dictionaries
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
    
    def _build_tool_specs(self, function_list: List[Union[str, BaseTool]]) -> List[Dict[str, Any]]:
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
            if isinstance(item, BaseTool):
                tool = item
            else:
                cls = name_to_cls.get(str(item))
                tool = None
                if cls is not None:
                    try:
                        tool = cls()
                    except Exception:
                        tool = None
            if tool is not None:
                specs.append({
                    'name': getattr(tool, 'name', str(item)),
                    'description': getattr(tool, 'description', 'No description'),
                    'parameters': getattr(tool, 'parameters', None)
                })
    
        return specs

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
    
    def _get_system_message(self, tool_specs: Optional[List[Dict[str, Any]]] = None) -> str:
        """Get the system message for the agent using ReAct paradigm"""
        tools_section = ""
        if tool_specs:
            lines = []
            for spec in tool_specs:
                desc = spec.get('description') or ''
                name = spec.get('name')
                lines.append(f"- {name}: {desc}")
            tools_section = "\n".join(lines)
        else:
            tools_section = (
                "- click: Click on elements by describing what you want to click\n"
                "- type: Type text into input fields by describing the field\n"
                "- press_key: Press specific keys (enter, delete, space)\n"
                "- scroll: Scroll the page in different directions\n"
                "- wait: Wait for 2 seconds (default) for page loading\n"
                "- content_analyzer: Analyze page content and images (results will be available for next step)\n"
                "- map_search: Navigate to Google Maps for geographical searches\n"
                "- goto_url: Navigate to the specific webpage, including tickets booking, car rental, flight booking, hotel booking, shopping, event search, shopping, map, youtube, food, travel guide, exchange dollars\n"
                # "- google_web_search: Google Search the web for information and get summarized results (results will be available for next step)\n"
                "- stop: Stop the task and provide final answer"
            )
        return f"""You are a GUI automation agent that can interact with web pages and applications using the ReAct (Reasoning and Acting) paradigm.

Your task is to:
1. Analyze the current state of the page
2. Think through what needs to be done (Reasoning)
3. Determine the appropriate action to take (Acting)
4. Use the available functions to perform the action
5. Continue until the task is complete

IMPORTANT: Always provide reasoning for your actions. Use the ReAct paradigm:
- First, think about what you need to do and why
- Then, choose the appropriate action with clear reasoning
- Finally, execute the action

WORKFLOW GUIDELINES:
- When you need to search for information: Directly type your search query, do not need to click the search bar
- After clicking an element, if you need to interact with it further (like typing), do so immediately
- Don't repeat the same action multiple times - if something doesn't work, try a different approach

Available actions:
{tools_section}

For each action, you must provide:
1. Reasoning: Why this action is necessary
2. Description: What element you're targeting
3. Execution: The actual action to perform

Always think before acting and provide clear reasoning for your decisions."""

    def next_action_custom(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: Dict[str, Any],
        model=None,
        args: argparse.Namespace = None,
        reflection_messages=None
    ) -> Action:
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
            messages = self._prepare_messages(trajectory, intent, meta_data)
            
            # Get functions for the agent
            functions = [func.function for func in self.function_map.values()]
            
            # Call the LLM with function calling
            responses = []
            for response in self.llm.chat(messages=messages, functions=functions, stream=False):
                        responses.append(response)
            print('*'*50, 'responses', '*'*50)
            print(responses)
            print('*'*50, 'responses', '*'*50)
                    
            # Extract page if available in meta_data or elsewhere; pass explicitly
            page_for_tools = meta_data.get('page')
                
            # Process the response
            action = self._process_response(responses, trajectory, page_for_tools, intent)
            print('*'*50, 'action', '*'*50)
            print(action)
            print('*'*50, 'action', '*'*50)
        
            # Check if action is None or empty
            if action is None or action.get('action_type') == ActionTypes.NONE:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}: Action is None, retrying...")
                    continue
                else:
                    print(f"Attempt {attempt + 1}: Action is None after {max_retries} attempts, returning None action")
                    return create_none_action()
            else:
                # Valid action found, return it
                return action
        
        # This should never be reached, but just in case
        return create_none_action()
    
    def _prepare_messages(self, trajectory: Trajectory, intent: str, meta_data: Dict[str, Any]) -> List[Dict]:
        """Prepare messages for the LLM with ReAct context"""
        messages = []
        
        # Add system message
        function_list = self._define_functions()
        tool_specs = self._build_tool_specs(function_list)
        messages.append({
            'role': 'system',
            'content': self._get_system_message(tool_specs)
        })
        
        # Add current intent with ReAct prompt
        current_task = intent
        
        # Add action history
        if meta_data and 'action_history' in meta_data:
            action_history = meta_data['action_history'][-5:]  # Last 5 actions
            if action_history:
                history_text = "Recent actions:\n" + "\n".join(action_history)
                messages.append({
                    'role': 'user',
                    'content': history_text
                })
        
        # Add analysis results context if available
        if self.last_analysis_result:
            # try:
            analysis_summary = self.last_analysis_result
            messages.append({
                'role': 'assistant',
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
                    ContentItem(text=f"Web search results: {web_search_summary}")
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
                            content_items.append(ContentItem(image=f"data:image/png;base64,{img_base64}"))
                        screenshot_files_to_delete.append(screenshot_path)
                
                messages.append({
                    'role': 'assistant',
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
                    'role': 'assistant',
                    'content': f"Web search results: {web_search_summary}"
                })
            
            # Clear the web search result after using it
            self.last_web_search_result = None
            self.last_web_search_screenshots = None
        
        # Add page goto results context if available
        if self.last_page_goto_result:
            page_goto_summary = f"Successfully navigated to {self.last_page_goto_name} at {self.last_page_goto_result}"
            messages.append({
                'role': 'assistant',
                'content': f"Page navigation results: {page_goto_summary}"
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
                            ContentItem(image=f'data:image/png;base64,{obs["image"]}'),
                            ContentItem(text=page_description)
                        ]
                    })
        messages.append({
            'role': 'user',
            'content': f"""Current task: {current_task}

Please follow the ReAct paradigm:
1. Think: Analyze what needs to be done
2. Reason: Explain why you're choosing this action
3. Act: Execute the appropriate action

IMPORTANT REMINDERS:
- If you need to search for information, directly type your search query, do not need to click the search bar
- Don't repeat the same action multiple times - try different approaches if something doesn't work
- Complete your task step by step, thinking through each action

What would you like to do next?"""
                })
        
        return messages
    
    def _process_response(self, responses: List[Message], trajectory: Trajectory, page: Optional[Any] = None, intent: Optional[str] = None) -> Action:
        """Process the LLM response and convert to Action"""
        
        # # Always run content analyzer to get current page analysis
        # # This ensures analysis results are available for the next step
        # tool = self.function_map.get('content_analyzer')
        # if tool and page:
        #     try:
        #         # Add trajectory context, page (if available), and LLM to kwargs
        #         content_analyzer_kwargs = {'page': page, 'analyze_images': False}
        #         content_analyzer_func_args = {
        #             'query': intent if intent else 'Analyze the current page content and structure',
        #             'reasoning': 'I need to analyze the page to understand the content and structure for better decision making.'
        #         }
        #         tool.llm = self.llm
        #         result = tool.call(json.dumps(content_analyzer_func_args), **content_analyzer_kwargs)
        #         self.last_analysis_result = result['page_content']
        #         current_url = page.url if hasattr(page, 'url') else None
        #         print('*'*50, 'Content Analysis Result', '*'*50)
        #         print(f"Analysis completed for: {intent if intent else 'current page'}")
        #         print(f"Page URL: {current_url}")
        #         print('*'*50, 'Content Analysis Result', '*'*50)
        #     except Exception as e:
        #         print(f"Error running content analyzer: {e}")
        #         self.last_analysis_result = None
            
        for response in responses:
            if 'function_call' in response and response['function_call']:
                func_name = response['function_call']['name']
                
                # Handle malformed JSON in function arguments
                try:
                    func_args = json.loads(response['function_call']['arguments'])
                except json.JSONDecodeError as e:
                    self.logger.warning(f"JSON decode error in function arguments: {e}")
                    self.logger.warning(f"Raw arguments: {response['function_call']['arguments']}")
                    
                    # Try to fix common JSON issues
                    raw_args = response['function_call']['arguments']
                    
                    # Fix trailing commas
                    raw_args = raw_args.replace(',}', '}').replace(',]', ']')
                    
                    # Fix missing quotes around keys
                    import re
                    raw_args = re.sub(r'(\w+):', r'"\1":', raw_args)
                    
                    # Try parsing again
                    try:
                        func_args = json.loads(raw_args)
                        self.logger.info("Successfully fixed JSON arguments")
                    except json.JSONDecodeError:
                        self.logger.error("Failed to fix JSON arguments, using empty dict")
                        return create_none_action()
                
                # # Log the reasoning
                # if 'reasoning' in func_args:
                #     self.logger.info(f"Agent reasoning: {func_args['reasoning']}")
                
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
                        tool.llm = self.llm
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
                        tool.llm = self.llm
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
                        tool.set_llm(self.llm)
                        
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
                        
                
        # If no function call, try to parse natural language responses
        for response in responses:
            if 'content' in response and response['content']:
                content = response['content']
                print(f"Parsing natural language content: {content}")
                
                # # First try to match action patterns in the content
                # action = self._parse_natural_language_action(content)
                # if action and action.get('action_type') != ActionTypes.NONE:
                #     print(f"Successfully parsed natural language action: {action}")
                #     return action
                
                # If pattern matching fails, use LLM to parse the content
                action = self._parse_natural_language_with_llm(content)
                if action and action.get('action_type') != ActionTypes.NONE:
                    print(f"Successfully parsed natural language action with LLM: {action}")
                    return action
        
        # If no function call and no natural language action found, return none action
        action = create_none_action()
        

        
        return action
    

    def _parse_natural_language_with_llm(self, content: str) -> Action:
        """Use LLM to parse natural language content and extract action information"""
        try:
            # Create a prompt for the LLM to parse the content
            system_prompt = """You are an expert at parsing natural language responses and converting them into structured actions for a GUI automation agent.

Available actions:
- click: Click on elements by describing what you want to click
- type: Type text into input fields by describing the field
- press_key: Press specific keys (enter, delete, space, etc.)
- scroll: Scroll the page in different directions (up, down, left, right)
- wait: Wait for a specified number of seconds
- stop: Stop the task and provide final answer
- map_search: Navigate to Google Maps for geographical searches
- content_analyzer: Analyze page content and images (results will be available for next step)

CRITICAL: You MUST respond with ONLY a valid JSON object. No other text, no explanations, no markdown formatting, just pure JSON.

Parse the given content and return a JSON object with the following structure:
{
    "action_type": "click|type|press_key|scroll|wait|stop",
    "description": "description of what to click",
    "text": "text to type (for type action)",
    "field_description": "description of the field (for type action)",
    "key": "key to press (for press_key action)",
    "direction": "scroll direction (for scroll action)",
    "seconds": "number of seconds (for wait action)",
    "answer": "final answer (for stop action)",
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
            response = self.llm.chat(messages=messages, stream=False)
            result = ""
            for resp in response:
                if hasattr(resp, 'content'):
                    result += resp.content
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
                return create_wait_action(
                    seconds=float(action_data.get('seconds', 2.0)),
                    reasoning=reasoning
                )
            elif action_type == 'google_web_search':
                return create_wait_action(
                    seconds=float(action_data.get('seconds', 2.0)),
                    reasoning=reasoning
                )
            elif action_type == 'content_analyzer':
                return create_wait_action(
                    seconds=float(action_data.get('seconds', 2.0)),
                    reasoning=reasoning
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
                        ContentItem(image=f'data:image/png;base64,{image_base64}'),
                        ContentItem(text="Please describe this web page screenshot. Include the main content, any visible buttons, forms, or interactive elements, and the overall purpose of the page.")
                    ]
                }
            ]
            
            # Get LLM response
            response = self.llm.chat(messages=messages, stream=False)
            description = ""
            for resp in response:
                if hasattr(resp, 'content'):
                    description += resp.content
                else:
                    description += str(resp)
            
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