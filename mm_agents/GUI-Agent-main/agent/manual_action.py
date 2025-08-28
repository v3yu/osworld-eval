"""
Manual Action Generator for GUI Agent

This module provides functionality to manually create actions for the web agent
that are compatible with the existing _process_response method.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from qwen_agent.llm.schema import Message
import base64
from PIL import Image
import io

logger = logging.getLogger(__name__)

def get_manual_action(trajectory, intent: str, meta_data) -> List[dict]:
    """
    Generate a manual action response that can be processed by _process_response.
    
    Args:
        trajectory: The current trajectory object
        intent: The current intent/task description
        
    Returns:
        List[dict]: A list containing a single message dictionary with function_call
    """
    
    print("\n" + "="*60)
    print("MANUAL ACTION MODE")
    print("="*60)
    print(f"Current Intent: {intent}")
    print(f"Current URL: {trajectory.current_url if hasattr(trajectory, 'current_url') else 'Unknown'}")
    print(f"Step: {len(meta_data['action_history'])}")
    
    image_obs_base64 = trajectory[-1]['observation']['image']
    image_obs = base64.b64decode(image_obs_base64)
    image_obs = Image.open(io.BytesIO(image_obs))
    image_obs.save('image_obs.png')
    
    # Show available actions
    print("\nAvailable Actions:")
    print("1. click - Click on elements")
    print("2. type - Type text into fields")
    print("3. press_key - Press specific keys")
    print("4. scroll - Scroll the page")
    print("5. wait - Wait for loading")
    print("6. stop - Stop and provide answer")
    print("7. map_search - Search on Google Maps")
    print("8. content_analyzer - Analyze page content")
    print("9. goto_url - Navigate to specific page")
    
    # Get user input
    while True:
        try:
            action_choice = input("\nEnter action number (1-9): ").strip()
            
            if action_choice == "1":  # click
                description = input("What do you want to click? (e.g., 'the search button'): ").strip()
                reasoning = input("Reasoning for this action: ").strip()
                
                return [{
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": "click",
                        "arguments": json.dumps({
                            "description": description,
                            "reasoning": reasoning
                        })
                    }
                }]
                
            elif action_choice == "2":  # type
                text = input("What text do you want to type? ").strip()
                field_description = input("Describe the field (e.g., 'search input field'): ").strip()
                reasoning = input("Reasoning for this action: ").strip()
                
                return [{
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": "type",
                        "arguments": json.dumps({
                            "text": text,
                            "field_description": field_description,
                            "reasoning": reasoning
                        })
                    }
                }]
                
            elif action_choice == "3":  # press_key
                key = input("Which key to press? (enter/delete/space/escape/tab): ").strip()
                reasoning = input("Reasoning for this action: ").strip()
                
                return [{
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": "press_key",
                        "arguments": json.dumps({
                            "key": key,
                            "reasoning": reasoning
                        })
                    }
                }]
                
            elif action_choice == "4":  # scroll
                direction = input("Scroll direction? (up/down/left/right): ").strip()
                reasoning = input("Reasoning for this action: ").strip()
                
                return [{
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": "scroll",
                        "arguments": json.dumps({
                            "direction": direction,
                            "reasoning": reasoning
                        })
                    }
                }]
                
            elif action_choice == "5":  # wait
                seconds = "2"
                reasoning = 'waiting for the page to load'
                
                return [{
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": "wait",
                        "arguments": json.dumps({
                            "seconds": float(seconds),
                            "reasoning": reasoning
                        })
                    }
                }]
                
            elif action_choice == "6":  # stop
                answer = input("What is the final answer? ").strip()
                reasoning = input("Reasoning for stopping: ").strip()
                
                return [{
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": "stop",
                        "arguments": json.dumps({
                            "answer": answer,
                            "reasoning": reasoning
                        })
                    }
                }]
                
            elif action_choice == "7":  # map_search
                query = input("What do you want to search on Google Maps? ").strip()
                reasoning = input("Reasoning for this action: ").strip()
                
                return [{
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": "map_search",
                        "arguments": json.dumps({
                            "query": query,
                            "reasoning": reasoning
                        })
                    }
                }]
                
            elif action_choice == "8":  # content_analyzer
                query = intent
                reasoning = input("Reasoning for this action: ").strip()
                
                return [{
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": "content_analyzer",
                        "arguments": json.dumps({
                            "query": query,
                            "reasoning": reasoning
                        })
                    }
                }]
                
            elif action_choice == "9":  # goto_url
                print("\nAvailable pages:")
                print("- book a hotel")
                print("- book a car") 
                print("- book a flight")
                print("- search on youtube")
                print("- search on twitter")
                print("- search some events")
                print("- find food")
                print("- travel guide")
                print("- exchange dollars")
                print("- shopping")
                
                page_name = input("Which page do you want to go to? ").strip()
                reasoning = input("Reasoning for this action: ").strip()
                
                return [{
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": "goto_url",
                        "arguments": json.dumps({
                            "page_name": page_name,
                            "reasoning": reasoning
                        })
                    }
                }]
                
            else:
                print("Invalid choice. Please enter a number between 1-9.")
                
        except KeyboardInterrupt:
            print("\n\nManual action cancelled. Returning none action.")
            return [{
                "role": "assistant",
                "content": "",
                "function_call": {
                    "name": "wait",
                    "arguments": json.dumps({
                        "seconds": 1.0,
                        "reasoning": "Manual action cancelled"
                    })
                }
            }]
        except Exception as e:
            print(f"Error in manual action: {e}")
            print("Please try again.")
            continue


def get_manual_action_simple(trajectory, intent: str) -> List[dict]:
    """
    A simpler version of manual action that just asks for the action type and basic parameters.
    
    Args:
        trajectory: The current trajectory object
        intent: The current intent/task description
        
    Returns:
        List[dict]: A list containing a single message dictionary with function_call
    """
    
    print(f"\nManual Action - Intent: {intent}")
    print("Quick action selection:")
    print("1. click")
    print("2. type") 
    print("3. scroll")
    print("4. wait")
    print("5. stop")
    
    action_map = {
        "1": ("click", {"description": "element", "reasoning": "manual click"}),
        "2": ("type", {"text": "text", "field_description": "field", "reasoning": "manual type"}),
        "3": ("scroll", {"direction": "down", "reasoning": "manual scroll"}),
        "4": ("wait", {"seconds": 2.0, "reasoning": "manual wait"}),
        "5": ("stop", {"answer": "manual stop", "reasoning": "manual stop"})
    }
    
    choice = input("Choose action (1-5): ").strip()
    
    if choice in action_map:
        action_name, default_args = action_map[choice]
        
        # Get basic input for the chosen action
        if action_name == "click":
            description = input("Click what? ") or default_args["description"]
            default_args["description"] = description
        elif action_name == "type":
            text = input("Type what? ") or default_args["text"]
            default_args["text"] = text
        elif action_name == "stop":
            answer = input("Final answer? ") or default_args["answer"]
            default_args["answer"] = answer
            
        return [{
            "role": "assistant",
            "content": "",
            "function_call": {
                "name": action_name,
                "arguments": json.dumps(default_args)
            }
        }]
    else:
        # Default to wait action
        return [{
            "role": "assistant",
            "content": "",
            "function_call": {
                "name": "wait",
                "arguments": json.dumps({
                    "seconds": 1.0,
                    "reasoning": "invalid manual action choice"
                })
            }
        }] 