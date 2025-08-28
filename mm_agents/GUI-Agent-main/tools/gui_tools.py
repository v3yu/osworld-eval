"""Custom GUI tools for the Qwen-Agent framework

This module originally depends on qwen_agent for BaseTool and register_tool.
To make OSWorld integration self-contained, we provide graceful fallbacks when
qwen_agent is not installed or the hardcoded path is unavailable.
"""
from typing import Dict, Any
import sys 
import json

# Try to import qwen_agent; if unavailable, define light-weight shims
try:
    sys.path.insert(0,'/lustre/scratch/users/guangyi.liu/agent/Qwen-Agent')
    from qwen_agent.tools import BaseTool
    from qwen_agent.tools.base import register_tool
except Exception:  # pragma: no cover - fallback for environments without qwen_agent
    class BaseTool:  # Minimal interface to satisfy downstream usage
        name: str = ""
        description: str = ""
        parameters: Dict[str, Any] | None = None
        def __init__(self, config: Dict[str, Any] | None = None):
            self.config = config or {}
        def call(self, args: str, **kwargs) -> str:
            return ""

    def register_tool(name: str):
        def decorator(cls):
            # No-op in fallback; simply attach name attribute if missing
            try:
                setattr(cls, 'name', name)
            except Exception:
                pass
            return cls
        return decorator


@register_tool('click')
class ClickTool(BaseTool):
    """Tool for clicking on elements in the GUI"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = 'click'
        self.description = 'Click on an element described by its appearance, text, or location'
        self.parameters = {
            'type': 'object',
            'properties': {
                'description': {
                    'type': 'string',
                    'description': 'The number label of the item you want to interact with, and the description of the element to click on'
                },
                'reasoning': {
                    'type': 'string',
                    'description': 'Reasoning for why this action is necessary'
                }
            },
            'required': ['description', 'reasoning']
        }
        
    def call(self, args: str, **kwargs) -> str:
        """Acknowledge click action - actual action creation happens in _process_response"""
        try:
            # Parse arguments for logging
            if isinstance(args, str):
                import json
                args = json.loads(args)
            
            description = args.get('description', '')
            reasoning = args.get('reasoning', '')
            
            # Just acknowledge the function call - action creation is handled by _process_response
            return f"Click function called with description: {description}, reasoning: {reasoning}"
            
        except Exception as e:
            return f"Error in click tool: {str(e)}"


@register_tool('type')
class TypeTool(BaseTool):
    """Tool for typing text into input fields"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = 'type'
        self.description = 'Type text into an input field described by its appearance or purpose'
        self.parameters = {
            'type': 'object',
            'properties': {
                'text': {
                    'type': 'string',
                    'description': 'Text to type into the input field'
                },
                'field_description': {
                    'type': 'string',
                    'description': 'The number label of the item you want to interact with, and the description of the input field to type into.'
                },
                'reasoning': {
                    'type': 'string',
                    'description': 'Reasoning for why this action is necessary'
                }
            },
            'required': ['text', 'field_description', 'reasoning']
        }
        
    def call(self, args: str, **kwargs) -> str:
        """Acknowledge type action - actual action creation happens in _process_response"""
        try:
            # Parse arguments for logging
            if isinstance(args, str):
                import json
                args = json.loads(args)
            
            text = args.get('text', '')
            field_description = args.get('field_description', '')
            reasoning = args.get('reasoning', '')
            
            # Just acknowledge the function call - action creation is handled by _process_response
            return f"Type function called with text: '{text}', field: {field_description}, reasoning: {reasoning}"
            
        except Exception as e:
            return f"Error in type tool: {str(e)}"


@register_tool('scroll')
class ScrollTool(BaseTool):
    """Tool for scrolling the page"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = 'scroll'
        self.description = 'Scroll the page up, down, left, or right'
        self.parameters = {
            'type': 'object',
            'properties': {
                'direction': {
                    'type': 'string',
                    'enum': ['up', 'down', 'left', 'right'],
                    'description': 'Direction to scroll'
                },
                'reasoning': {
                    'type': 'string',
                    'description': 'Reasoning for why this action is necessary'
                }
            },
            'required': ['direction', 'reasoning']
        }
        
    def call(self, args: str, **kwargs) -> str:
        """Acknowledge scroll action - actual action creation happens in _process_response"""
        try:
            # Parse arguments for logging
            if isinstance(args, str):
                import json
                args = json.loads(args)
            
            direction = args.get('direction', 'down')
            reasoning = args.get('reasoning', '')
            
            # Just acknowledge the function call - action creation is handled by _process_response
            return f"Scroll function called with direction: {direction}, reasoning: {reasoning}"
            
        except Exception as e:
            return f"Error in scroll tool: {str(e)}"


@register_tool('wait')
class WaitTool(BaseTool):
    """Tool for waiting a specified amount of time"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = 'wait'
        self.description = 'Wait for a specified amount of time (default 2 seconds)'
        self.parameters = {
            'type': 'object',
            'properties': {
                'reasoning': {
                    'type': 'string',
                    'description': 'Reasoning for why this action is necessary'
                }
            },
            'required': ['reasoning']
        }
        
    def call(self, args: str, **kwargs) -> str:
        """Acknowledge wait action - actual action creation happens in _process_response"""
        try:
            # Parse arguments for logging
            if isinstance(args, str):
                import json
                args = json.loads(args)
            
            reasoning = args.get('reasoning', '')
            
            # Just acknowledge the function call - action creation is handled by _process_response
            return f"Wait function called with reasoning: {reasoning}"
            
        except Exception as e:
            return f"Error in wait tool: {str(e)}"


@register_tool('press_key')
class PressKeyTool(BaseTool):
    """Tool for pressing specific keys"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = 'press_key'
        self.description = 'Press a specific key (enter, delete, space)'
        self.parameters = {
            'type': 'object',
            'properties': {
                'key': {
                    'type': 'string',
                    'enum': ['enter', 'delete', 'space'],
                    'description': 'The key to press'
                },
                'reasoning': {
                    'type': 'string',
                    'description': 'Reasoning for why this action is necessary'
                }
            },
            'required': ['key', 'reasoning']
        }
        
    def call(self, args: str, **kwargs) -> str:
        """Acknowledge press key action - actual action creation happens in _process_response"""
        try:
            # Parse arguments for logging
            if isinstance(args, str):
                import json
                args = json.loads(args)
            
            key = args.get('key', 'enter')
            reasoning = args.get('reasoning', '')
            
            # Just acknowledge the function call - action creation is handled by _process_response
            return f"Press key function called with key: {key}, reasoning: {reasoning}"
            
        except Exception as e:
            return f"Error in press key tool: {str(e)}"


@register_tool('stop')
class StopTool(BaseTool):
    """Tool for stopping the agent and providing an answer"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = 'stop'
        self.description = 'Stop the agent and provide an answer to the task'
        self.parameters = {
            'type': 'object',
            'properties': {
                'answer': {
                    'type': 'string',
                    'description': 'Final answer to the task'
                },
                'reasoning': {
                    'type': 'string',
                    'description': 'Reasoning for why the task is complete'
                }
            },
            'required': ['answer', 'reasoning']
        }
        
    def call(self, args: str, **kwargs) -> str:
        """Acknowledge stop action - actual action creation happens in _process_response"""
        try:
            # Parse arguments for logging
            if isinstance(args, str):
                import json
                args = json.loads(args)
            
            answer = args.get('answer', '')
            reasoning = args.get('reasoning', '')
            
            # Just acknowledge the function call - action creation is handled by _process_response
            return f"Stop function called with answer: {answer}, reasoning: {reasoning}"
            
        except Exception as e:
            return f"Error in stop tool: {str(e)}" 
        
@register_tool('goto_url')
class PageGotoTool(BaseTool):
    """Tool for navigating to specific pages based on user intent"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = 'goto_url'
        self.description = 'Navigate to the specific url, including tickets booking, car rental, flight booking, hotel booking, shopping, event search, shopping, map, youtube, food, travel guide, exchange dollars'
        self.parameters = {
            'type': 'object',
            'properties': {
                'page_name': {
                    'type': 'string',
                    'description': 'The page name to navigate to, must be one of: tickets booking, car rental, flight booking, hotel booking, shopping, event search, shopping, map, youtube, food, travel guide, exchange dollars'
                },
                'reasoning': {
                    'type': 'string',
                    'description': 'Reasoning for why this navigation is necessary'
                }
            },
            'required': ['page_name', 'reasoning']
        }
        
        # Define the mapping of page names to URLs (restricted set)
        self.page_urls = {
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
            'wiki': 'https://www.wikipedia.org/',
        }
    
    def call(self, args: str) -> str:
        """Return the target URL for the specified page based on user intent"""
        if isinstance(args, str):
            args = json.loads(args)
        
        reasoning = args.get('reasoning', '')
        page_name = args.get('page_name', '').lower()
        if 'car' in page_name:
            page_name = 'car'
        elif 'ticket' in page_name:
            page_name = 'ticket'
        elif 'flight' in page_name:
            page_name = 'flight'
        elif 'hotel' in page_name:
            page_name = 'hotel'
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
        elif 'wiki' in page_name:
            page_name = 'wiki'
            
        if page_name in self.page_urls:
            return self.page_urls[page_name.lower()]
        else:
            return page_name.lower()