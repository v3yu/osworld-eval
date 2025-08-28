"""
Task decomposition utilities for web navigation agents.
Handles query decomposition, answer extraction, and context building.
"""

import json
import re
import logging
from typing import List, Tuple, Dict, Any
from pathlib import Path

from browser_env import Trajectory, Action

logger = logging.getLogger(__name__)


def decompose_query(
    intent: str,
    decomposition_instruction_path: str,
    tokenizer,
    args,
    provider: str = "custom",
    max_sub_queries: int = 5
) -> List[str]:
    """
    Decompose a complex query into smaller sub-queries.
    
    Args:
        intent: The original complex query
        decomposition_instruction_path: Path to decomposition instruction JSON
        tokenizer: Tokenizer for the model
        args: Arguments object
        provider: LLM provider ("openai" or "custom")
        max_sub_queries: Maximum number of sub-queries to generate (default: 5)
    
    Returns:
        List of sub-queries
    """
    logger.info(f"[Task Decomposition] Starting decomposition for: {intent}")
    
    # try:
    from agent.prompts.prompt_constructor import DecompositionPromptConstructor
    
    # Create decomposition constructor
    decomp_constructor = DecompositionPromptConstructor(
        decomposition_instruction_path,
        args.lm_config,
        tokenizer,
        args,
    )
    
    # Construct decomposition prompt
    decomp_prompt = decomp_constructor.construct(
        trajectory=[],
        intent=intent,
        meta_data={}
    )
    
    # Replace [limit_number] placeholder with actual max_sub_queries value
    if isinstance(decomp_prompt, list):
        for message in decomp_prompt:
            if "content" in message:
                if isinstance(message["content"], list):
                    for content_item in message["content"]:
                        if isinstance(content_item, dict) and "text" in content_item:
                            content_item["text"] = content_item["text"].replace("[limit_number]", str(max_sub_queries))
    elif isinstance(decomp_prompt, str):
        decomp_prompt = decomp_prompt.replace("[limit_number]", str(max_sub_queries))
    print("--------------------------------DECOMPOSITION PROMPT--------------------------------")
    print(decomp_prompt)
    # Get decomposition from LLM using the existing model
    from llms import get_llm_backend
    llm = get_llm_backend(args, args.loaded_model)
    decomp_response = llm.generate(decomp_prompt)
    print("--------------------------------DECOMPOSITION RESPONSE--------------------------------")
    print(decomp_response)
    # Extract sub-queries
    decomp_result = decomp_constructor.extract_decomposition(decomp_response)
    sub_queries = decomp_result["sub_queries"]
    reasoning = decomp_result["reasoning"]
    
    # Limit the number of sub-queries
    if len(sub_queries) > max_sub_queries:
        logger.info(f"[Task Decomposition] Limiting sub-queries from {len(sub_queries)} to {max_sub_queries}")
        sub_queries = sub_queries[-max_sub_queries:]
    
    print("--------------------------------DECOMPOSITION RESULT--------------------------------")
    print(sub_queries)
    
    logger.info(f"[Task Decomposition] Generated {len(sub_queries)} sub-queries:")
    for i, sub_query in enumerate(sub_queries, 1):
        logger.info(f"[Task Decomposition] {i}. {sub_query}")
    logger.info(f"[Task Decomposition] Reasoning: {reasoning}")
    
    return sub_queries
    
    # except Exception as e:
    #     logger.warning(f"[Task Decomposition] Failed to decompose task: {str(e)}")
    #     logger.warning(f"[Task Decomposition] Falling back to original intent: {intent}")
    #     return [intent]  # Fallback to original intent


def extract_answer_from_response(trajectory: Trajectory, current_intent: str, sub_query_idx: int, args) -> str:
    """
    Extract the answer from a completed sub-query trajectory.
    
    Args:
        trajectory: The trajectory containing the sub-query execution
        current_intent: The current sub-query being processed
        sub_query_idx: Index of the current sub-query
    
    Returns:
        Extracted answer string
    """
    # try:
    # Get the model's full response from the last action
    last_action = None
    for item in reversed(trajectory):
        if isinstance(item, dict) and 'action_type' in item:
            last_action = item
            break
    
    if last_action:
        full_response = last_action.get('text', None)
        if not full_response:
            return ''
        
        # Extract answer from the full response
        answer_match = re.search(r'ANSWER:\s*(.*?)(?:\n|$)', full_response, re.DOTALL)
        if answer_match:
            sub_query_answer = answer_match.group(1).strip()
            logger.info(f"[Sub-query {sub_query_idx + 1}] Answer: {sub_query_answer}")
            return sub_query_answer
        else:
            # If no ANSWER: found, use the entire response
            sub_query_answer = full_response.strip()
            logger.info(f"[Sub-query {sub_query_idx + 1}] Answer (from full response): {sub_query_answer}")
            return sub_query_answer
    else:
        # Fallback: create a generic answer
        generic_answer = f"Completed sub-query: {current_intent}"
        logger.info(f"[Sub-query {sub_query_idx + 1}] Generic answer: {generic_answer}")
        return generic_answer
            
    # except Exception as e:
    #     logger.warning(f"[Sub-query {sub_query_idx + 1}] Failed to extract answer: {str(e)}")
    #     # Add a fallback answer
    #     fallback_answer = f"Error processing: {current_intent}"
    #     return fallback_answer


def build_context_from_answers(sub_query_answers: List[Tuple[str, str]]) -> str:
    """
    Build context string from previous sub-query answers.
    
    Args:
        sub_query_answers: List of (sub_query, answer) tuples
    
    Returns:
        Formatted context string
    """
    if not sub_query_answers:
        return ""
    
    context_parts = []
    for i, (prev_query, prev_answer) in enumerate(sub_query_answers, 1):
        context_parts.append(f"Sub-query {i}: {prev_query} → Answer: {prev_answer}")
    
    context_info = "Previous sub-query answers:\n" + "\n".join(context_parts) + "\n\n"
    return context_info


def enhance_intent_with_context(
    current_intent: str,
    sub_query_answers: List[Tuple[str, str]],
    use_task_decomposition: bool = True
) -> str:
    """
    Enhance the current intent with context from previous sub-query answers.
    
    Args:
        current_intent: The current sub-query
        sub_query_answers: List of (sub_query, answer) tuples from previous sub-queries
        use_task_decomposition: Whether task decomposition is enabled
    
    Returns:
        Enhanced intent with context
    """
    if sub_query_answers and use_task_decomposition:
        context_info = build_context_from_answers(sub_query_answers)
        enhanced_intent = context_info + f"Current sub-query: {current_intent}"
        logger.info(f"Enhanced intent with {len(sub_query_answers)} previous answers")
        return enhanced_intent
    else:
        return current_intent


def process_task_decomposition(
    intent: str,
    sub_queries: List[str],
    trajectory: Trajectory,
    args,
    use_task_decomposition: bool = True
) -> List[Tuple[str, str]]:
    """
    Process task decomposition and extract answers from each sub-query.
    
    Args:
        intent: Original intent
        sub_queries: List of sub-queries to process
        trajectory: Current trajectory
        args: Arguments object
        use_task_decomposition: Whether task decomposition is enabled
    
    Returns:
        List of (sub_query, answer) tuples
    """
    sub_query_answers = []
    
    for sub_query_idx, current_intent in enumerate(sub_queries):
        logger.info(f"[Sub-query {sub_query_idx + 1}/{len(sub_queries)}] Processing: {current_intent}")
        
        # Extract answer from completed sub-query
        if use_task_decomposition:
            sub_query_answer = extract_answer_from_response(trajectory, current_intent, sub_query_idx, args)
            sub_query_answers.append((current_intent, sub_query_answer))
    
    # Log final context that was available to the last sub-query
    if use_task_decomposition and len(sub_query_answers) > 0:
        logger.info(f"[Task Decomposition] Final sub-query had access to {len(sub_query_answers)} previous answers")
        for i, (query, answer) in enumerate(sub_query_answers, 1):
            logger.info(f"[Task Decomposition] Sub-query {i}: {query} → {answer}")
    
    return sub_query_answers 