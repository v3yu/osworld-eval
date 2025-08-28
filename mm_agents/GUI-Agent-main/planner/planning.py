"""
Planning module for generating logical plans using GPT-4o
"""
import argparse
from typing import Dict, Any, List, Optional
from .model import GPT4o


class Planning:
    """
    A planning class that uses GPT-4o to generate logical and concise plans
    
    This class provides methods for generating initial plans and managing
    planning workflows for web automation tasks.
    """
    
    def __init__(self, args: argparse.Namespace):
        """
        Initialize the planning system
        
        Args:
            args: Arguments object containing model configuration
        """
        self.args = args
        self.gpt4o = GPT4o(args)
        
    def generate_initial_plan(self, query: str, start_url: str) -> str:
        """
        Generate an initial plan for a given query and start URL
        
        Args:
            query: The user's query or task description
            start_url: The starting URL for the task
            
        Returns:
            A markdown-formatted plan
        """
        system_prompt = """You are an expert web automation planner. Your task is to create logical, concise, and actionable plans for web-based tasks.

## Planning Guidelines:
1. **Be Logical**: Break down complex tasks into clear, sequential steps
2. **Be Concise**: Keep each step brief but informative
3. **Be Actionable**: Each step should be something that can be executed
4. **Consider Navigation**: Think about how to move from one page to another
5. **Handle Edge Cases**: Consider what might go wrong and how to handle it
6. **Use Markdown**: Format your response in clean markdown

## Plan Structure:
- Start with a brief overview of the task
- List numbered steps in logical order
- Include decision points where needed
- End with expected outcomes

## Example:
```markdown
# Task Description: How to Find the Date Difference Between Two APEC Events "Tourism Ministers joint statement" and "Tourism Ministers Meeting"

## Plan
1. **Start from the official website**  
   → [https://www.apec.org](https://www.apec.org)

2. **Use the search button** (top-right corner)  
   - Search: `"Tourism Ministers joint statement"`  
   - Search: `"Minister Galdo"`

3. **Find and open both items**  
   - Locate the **publication or event date** for each.

4. **Calculate the difference in days**  
   - Subtract the earlier date from the later date.
   
## Expected Outcome
We expect to search for accurate information and find the date difference between the two events.
```"""

        user_prompt = f"""Please create a logical and concise plan for the following task:

**Query:** {query}
**Starting URL:** {start_url}

Generate a plan that:
1. Starts from the given URL
2. Breaks down the task into clear, executable steps
3. Considers navigation between pages
4. Leads to the desired outcome

Format your response in clean markdown with clear sections and numbered steps."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        plan = self.gpt4o.chat(
            messages=messages,
            temperature=0.3,  # Lower temperature for more focused planning
            max_tokens=1500
        )
        return plan
    
    def step_plan(self, initial_plan: str, current_url: str, screenshot_base64: str, previous_actions: List[str], tool_specs: List[Dict[str, Any]]) -> str:
        """
        Generate suggestions for the current step based on the initial plan and current context
        
        Args:
            initial_plan: The original plan generated for the task
            current_url: The current URL the agent is on
            screenshot_base64: Base64 encoded screenshot of the current page
            previous_actions: List of the last 3 actions taken
            
        Returns:
            Suggestions for the current step in markdown format
        """
        lines = []
        for spec in tool_specs:
            desc = spec.get('description') or ''
            name = spec.get('name')
            lines.append(f"- {name}: {desc}")
        tools_section = "\n".join(lines)
        
        system_prompt = f"""You are an expert web automation assistant. Your task is to analyze the current situation and provide specific, actionable suggestions for the next step.

## Analysis Guidelines:
1. **Review the Plan**: Understand what the overall plan is trying to achieve
2. **Assess Current State**: Analyze the current URL and page content
3. **Consider Previous Actions**: Learn from what has already been done
4. **Identify Next Step**: Determine what should be done next
5. **Provide Specific Actions**: Give concrete, executable suggestions

## Aviliable Actions
{tools_section}

## Response Format (Do not include any other section):
```markdown
# Current Step Analysis

## Current Situation
- **URL**: [current URL]
- **Page Type**: [what type of page this appears to be]
- **Key Elements**: [important elements visible on the page]

## Progress Assessment
- **Plan Progress**: [how much of the plan has been completed]
- **Previous Actions**: [summary of recent actions taken]
- **Current Position**: [where we are in the plan]

## Next Step Suggestions
**Primary Action**: [most likely next action]
   - **Reasoning**: [why this action makes sense]
   - **Expected Outcome**: [what should happen]

```"""

        # Format previous actions for the prompt
        previous_actions_text = "\n".join([f"- {action}" for action in previous_actions]) if previous_actions else "- No previous actions"

        user_prompt = f"""Please analyze the current situation and provide specific suggestions for the next step.

**Initial Plan:**
{initial_plan}

**Current Context:**
- **Current URL**: {current_url}
- **Previous Actions**: 
{previous_actions_text}
NOTE: In your plan DO NOT REPEAT previous actions! If you observe repeat in previous actions, do UPDATE your STEP SUGGESTION!

**Current Page Screenshot**: [Image provided]

Based on the initial plan, current URL, and previous actions, what should be the next step? Provide specific, actionable suggestions that will help progress toward the goal.

Consider:
1. What step in the plan should we be on now?
2. What elements on the current page are relevant?
3. What action would best advance toward the goal?
4. What alternatives should we consider if the primary action fails?"""

        # Create content with image
        content = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{screenshot_base64}"
                }
            }
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        
        step_suggestions = self.gpt4o.chat(
            messages=messages,
            temperature=0.4,  # Balanced creativity and focus
            max_tokens=1200
        )
        return step_suggestions
               
def create_planning_system(args: argparse.Namespace) -> Planning:
    """
    Factory function to create a planning system instance
    
    Args:
        args: Arguments object containing model configuration
        
    Returns:
        Planning system instance
    """
    return Planning(args)
