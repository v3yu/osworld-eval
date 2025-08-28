# Manual Action Functionality

This module provides manual action generation for the GUI Agent, allowing users to manually control the agent's actions during execution.

## Overview

The manual action functionality allows you to:
- Manually specify actions for the agent instead of relying on LLM decisions
- Debug and test specific action sequences
- Override agent behavior when needed
- Create controlled test scenarios

## Files

- `manual_action.py` - Main implementation of manual action functions
- `test_manual_action.py` - Test script to verify functionality

## Usage

### 1. Enable Manual Action Mode

Add the `--manual_action` flag when running the agent:

```bash
python run.py --manual_action --domain shopping
```

### 2. Available Actions

When manual action mode is enabled, you'll be prompted to choose from these actions:

1. **click** - Click on elements
   - Parameters: description, reasoning
   
2. **type** - Type text into fields
   - Parameters: text, field_description, reasoning
   
3. **press_key** - Press specific keys
   - Parameters: key (enter/delete/space/escape/tab), reasoning
   
4. **scroll** - Scroll the page
   - Parameters: direction (up/down/left/right), reasoning
   
5. **wait** - Wait for loading
   - Parameters: seconds, reasoning
   
6. **stop** - Stop and provide answer
   - Parameters: answer, reasoning
   
7. **map_search** - Search on Google Maps
   - Parameters: query, reasoning
   
8. **content_analyzer** - Analyze page content
   - Parameters: query, reasoning
   
9. **goto_url** - Navigate to specific page
   - Parameters: page_name, reasoning

### 3. Available Pages for goto_url

When selecting the `goto_url` action, you can choose from these predefined pages:

- book a hotel
- book a car
- book a flight
- search on youtube
- search on twitter
- search some events
- find food
- travel guide
- exchange dollars
- shopping

## Function Details

### `get_manual_action(trajectory, intent)`

The main function that provides a full interactive interface for manual action selection.

**Parameters:**
- `trajectory`: The current trajectory object containing state information
- `intent`: The current task/intent description

**Returns:**
- `List[Message]`: A list containing a single message with function_call that can be processed by `_process_response`

### `get_manual_action_simple(trajectory, intent)`

A simplified version that provides quick action selection with minimal prompts.

**Parameters:**
- `trajectory`: The current trajectory object containing state information
- `intent`: The current task/intent description

**Returns:**
- `List[Message]`: A list containing a single message with function_call

## Integration with Agent

The manual action functionality is integrated into the agent's `next_action_custom` method:

```python
# Check if manual action mode is enabled
if self.args.manual_action:
    responses = get_manual_action(trajectory, intent)
else:
    # Call the LLM with function calling
    responses = []
    for response in self.llm.chat(messages=messages, functions=functions, stream=False):
        responses.append(response)
```

## Response Format

The manual action functions return responses in the same format expected by `_process_response`:

```python
[Message(
    role="assistant",
    content="",
    function_call={
        "name": "click",
        "arguments": json.dumps({
            "description": "the search button",
            "reasoning": "Need to click the search button to submit the query"
        })
    }
)]
```

## Testing

Run the test script to verify the functionality:

```bash
cd GUI-Agent
python test_manual_action.py
```

The test script will:
1. Test the full manual action function
2. Test the simple manual action function
3. Verify that responses are in the correct format
4. Display the generated function calls

## Error Handling

The manual action functions include error handling for:
- Invalid user input
- Keyboard interrupts (Ctrl+C)
- JSON serialization errors
- Missing parameters

## Example Usage

```bash
# Run agent with manual action mode
python run.py --manual_action --domain shopping --max_steps 5

# The agent will prompt you for each action:
# ============================================================
# MANUAL ACTION MODE
# ============================================================
# Current Intent: Find information about iPhone 15
# Current URL: https://example.com
# Step: 0
# 
# Available Actions:
# 1. click - Click on elements
# 2. type - Type text into fields
# ...
# 
# Enter action number (1-9): 2
# What text do you want to type? iPhone 15
# Describe the field (e.g., 'search input field'): search box
# Reasoning for this action: Need to search for iPhone 15 information
```

## Benefits

1. **Debugging**: Easily test specific action sequences
2. **Control**: Override agent decisions when needed
3. **Testing**: Create reproducible test scenarios
4. **Learning**: Understand how different actions affect the agent's behavior
5. **Development**: Test new features without relying on LLM decisions

## Limitations

1. **Manual Input**: Requires user interaction for each action
2. **No Automation**: Cannot run completely automated tests
3. **Time Consuming**: Slower than automated execution
4. **Human Error**: Subject to user input mistakes

## Future Enhancements

Potential improvements could include:
- Batch action input
- Action sequence recording and replay
- Integration with test frameworks
- Automated action validation
- Action templates for common scenarios 