"""Fallback answer generation for the GUI Agent"""
import logging
from typing import Dict, Any, List
from browser_env import Trajectory


def generate_fallback_answer(question: str, trajectory: Trajectory, args, num_screenshots: int = 5) -> Dict[str, Any]:
    """
    Generate a fallback answer when the agent is early stopped
    
    Args:
        question: The original question/intent
        trajectory: The current trajectory
        args: Arguments object containing model and other settings
        num_screenshots: Number of latest screenshots to use
        
    Returns:
        Dictionary containing the generated answer
    """
    logger = logging.getLogger("logger")
    
    try:
        # Extract the latest screenshots from trajectory
        screenshots = _extract_latest_screenshots(trajectory, num_screenshots)
        
        if not screenshots:
            logger.warning("No screenshots found in trajectory for fallback answer")
            return {"answer": "Unable to generate fallback answer due to insufficient visual information."}
        
        # Create a prompt for fallback answer generation
        prompt = _create_fallback_prompt(question, screenshots)
        
        # Generate answer using the loaded model
        if hasattr(args, 'loaded_model') and args.loaded_model is not None:
            answer = _generate_answer_with_model(prompt, args)
        else:
            # Fallback to a simple response
            answer = _generate_simple_fallback_answer(question, trajectory)
        
        logger.info(f"Generated fallback answer: {answer}")
        return {"answer": answer}
        
    except Exception as e:
        logger.error(f"Error generating fallback answer: {e}")
        return {"answer": "Failed to generate fallback answer due to an error."}


def _extract_latest_screenshots(trajectory: Trajectory, num_screenshots: int) -> List[Any]:
    """
    Extract the latest screenshots from the trajectory
    
    Args:
        trajectory: The trajectory to extract from
        num_screenshots: Number of screenshots to extract
        
    Returns:
        List of screenshot data
    """
    screenshots = []
    
    # Go through trajectory in reverse to get latest screenshots
    for item in reversed(trajectory):
        if isinstance(item, dict) and 'observation' in item:
            obs = item['observation']
            if isinstance(obs, dict) and 'image' in obs:
                screenshots.append(obs['image'])
                if len(screenshots) >= num_screenshots:
                    break
    
    return list(reversed(screenshots))  # Return in chronological order


def _create_fallback_prompt(question: str, screenshots: List[Any]) -> str:
    """
    Create a prompt for fallback answer generation
    
    Args:
        question: The original question
        screenshots: List of screenshots
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""Based on the following question and the provided screenshots, please provide a comprehensive answer.

Question: {question}

The agent was unable to complete the task through direct interaction, but we have captured the current state of the interface. Please analyze the screenshots and provide the best possible answer based on the available information.

Consider:
1. What information is visible in the screenshots?
2. What might be the answer to the question based on the visible content?
3. Are there any clues or hints in the interface that could help answer the question?

Please provide a detailed answer based on the available visual information."""

    return prompt


def _generate_answer_with_model(prompt: str, args) -> str:
    """
    Generate answer using the loaded model
    
    Args:
        prompt: The prompt to use
        args: Arguments object with model information
        
    Returns:
        Generated answer string
    """
    try:
        # This is a placeholder - you'll need to implement the actual model call
        # based on your specific model interface
        if hasattr(args.loaded_model, 'generate'):
            # Example for a model with generate method
            response = args.loaded_model.generate(prompt)
            return response
        else:
            # Fallback for other model types
            return _generate_simple_fallback_answer("", [])
    except Exception as e:
        logging.getLogger("logger").error(f"Error calling model for fallback answer: {e}")
        return _generate_simple_fallback_answer("", [])


def _generate_simple_fallback_answer(question: str, trajectory: Trajectory) -> str:
    """
    Generate a simple fallback answer when model is not available
    
    Args:
        question: The original question
        trajectory: The trajectory
        
    Returns:
        Simple fallback answer
    """
    return f"I was unable to complete the task '{question}' through direct interaction. The agent reached its maximum step limit or encountered an error. Please try again or rephrase your request." 