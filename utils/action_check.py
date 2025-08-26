"""Action self-check and retry functionality for the GUI Agent"""
import logging
from typing import Callable, Any


def action_self_check(gen_action: Callable, intent: str, page, trajectory, max_retries: int = 3, repeat_threshold: int = 3):
    """
    Perform action self-check and retry if needed
    
    Args:
        gen_action: Function to generate action
        intent: Current intent
        page: Browser page object
        trajectory: Current trajectory
        max_retries: Maximum number of retries
        repeat_threshold: Threshold for repeating actions
        
    Returns:
        Generated action
    """
    logger = logging.getLogger("logger")
    
    for attempt in range(max_retries):
        # try:
        action = gen_action(intent)
        
        # Check if action is valid
        if action is None or (isinstance(action, dict) and action.get('action_type') == ''):
            logger.warning(f"Generated invalid action on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                continue
            else:
                logger.error("Failed to generate valid action after all retries")
                return action
        
        # Check for repeating actions
        if _is_repeating_action(trajectory, action, repeat_threshold):
            # logger.warning(f"Detected repeating action on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                # Add error feedback to encourage different action
                error_message = "The last action was repeated multiple times. Please try a different approach."
                action = gen_action(intent, error_message)
            else:
                logger.error("Failed to generate non-repeating action after all retries")
                return action
        
        return action
        
        # except Exception as e:
        #     logger.error(f"Error generating action on attempt {attempt + 1}: {e}")
        #     if attempt < max_retries - 1:
        #         continue
        #     else:
        #         logger.error("Failed to generate action after all retries")
        #         raise
    
    return None


def _is_repeating_action(trajectory, action, threshold: int) -> bool:
    """
    Check if the action is being repeated too many times
    
    Args:
        trajectory: Current trajectory
        action: Current action
        threshold: Threshold for considering action as repeating
        
    Returns:
        True if action is repeating too much
    """
    # Extract actions from trajectory
    actions = [item for item in trajectory if isinstance(item, dict) and item.get('action_type', '') != '']
    
    if len(actions) < threshold:
        return False
    
    # Check if the last 'threshold' actions are the same
    recent_actions = actions[-threshold:]
    
    # Compare current action with recent actions
    for recent_action in recent_actions:
        if not _actions_equivalent(action, recent_action):
            return False
    
    return True


def _actions_equivalent(action1, action2) -> bool:
    """
    Check if two actions are equivalent
    
    Args:
        action1: First action
        action2: Second action
        
    Returns:
        True if actions are equivalent
    """
    if not isinstance(action1, dict) or not isinstance(action2, dict):
        return False
    
    # Compare action types
    if action1.get('action_type') != action2.get('action_type'):
        return False
    
    # For specific action types, compare additional properties
    action_type = action1.get('action_type')
    
    if action_type in ['click', 'type']:
        # Compare coordinates for click actions
        if 'coord' in action1 and 'coord' in action2:
            return action1['coord'] == action2['coord']
        # Compare text for type actions
        elif 'text' in action1 and 'text' in action2:
            return action1['text'] == action2['text']
    
    return True 