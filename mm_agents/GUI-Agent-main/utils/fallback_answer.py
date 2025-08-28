"""Fallback answer generation for the GUI Agent"""
import logging
from typing import Dict, Any, List
from browser_env import Trajectory
import sys
sys.path.insert(0, '/lustre/scratch/users/guangyi.liu/agent/Qwen-Agent')
from qwen_agent.llm.schema import ContentItem

def generate_fallback_answer(question: str, trajectory: Trajectory, model, num_screenshots: int = 5) -> Dict[str, Any]:
    """
    Generate a fallback answer when the agent is early stopped
    
    Args:
        question: The original question/intent
        trajectory: The current trajectory
        model: The model to use
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
        content_items = _create_fallback_prompt(question, screenshots)
        
        # Generate answer using the loaded model
        answer = _generate_answer_with_model(model, content_items)
        
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


def _create_fallback_prompt(question: str, screenshots: List[Any]) -> List[ContentItem]:
    """
    Create a prompt for fallback answer generation
    
    Args:
        question: The original question
        screenshots: List of screenshots
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""Based on the following question and the provided screenshots, please provide a concise answer.

Question: {question}

The agent was unable to complete the task through direct interaction, but we have captured the current state of the interface. Please analyze the screenshots and provide the best possible answer based on the available information.

Consider:
1. What information is visible in the screenshots?
2. What might be the answer to the question based on the visible content?
3. Are there any clues or hints in the interface that could help answer the question?

Please provide a concise answer based on the available information."""

    # Create content items for the user message
    content_items = []

    if screenshots:
        for screenshot in screenshots:
            content_items.append({'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{screenshot}'}}
            ]})

    content_items.append({'role': 'user', 'content': prompt})

    return content_items



def _generate_answer_with_model(model, content_items: List[Dict[str, Any]]) -> str:
    """
    Generate answer using the loaded model

    Args:
        content_items: List of content items
        model: The model to use

    Returns:
        Generated answer string
    """
    
    # Create messages in the format expected by the LLM
    messages = [
        {
            'role': 'system',
            'content': 'You are a helpful assistant that provides accurate and concise answers to questions based on the provided information and screenshots.'
        }]
    messages.extend(content_items)
    
    # Call the LLM using the same pattern as in agent.py
    response = model.chat(messages=messages, stream=False)
    if isinstance(response, list):
        response = response[0]
    # Extract the response content
    if hasattr(response, 'content'):
        result = response.content
    elif isinstance(response, dict) and 'content' in response:
        # Handle dictionary response format
        result = response['content']
    else:
        result += str(response)
    
    # Clean up the response if it contains dictionary-like content
    result = result.replace("\"text\": \"{'role': 'assistant', 'content': '", "")
    result = result.replace("'}\"", "")
    result = result.replace("{'role': 'assistant', 'content': '", "")
    result = result.replace("'}", "")
    
    return result.strip()

