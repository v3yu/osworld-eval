# Function Call Agent with ReAct Paradigm

This module implements a function call agent using the Qwen-Agent framework with vLLM support for qwen2.5-instruct-VL, following the ReAct (Reasoning and Acting) paradigm.

## Features

- **ReAct Paradigm**: Uses reasoning before acting approach
- **Function Calling**: Uses Qwen-Agent's function calling capabilities
- **vLLM Integration**: Configured to work with vLLM server running qwen2.5-instruct-VL
- **Action Validation**: Built-in action validation after every generated action
- **Multimodal Support**: Can process images and text for GUI interactions

## ReAct Paradigm

The agent follows the ReAct (Reasoning and Acting) paradigm:

1. **Think**: Analyze what needs to be done
2. **Reason**: Explain why you're choosing this action
3. **Act**: Execute the appropriate action

For each action, the agent provides:
- **Reasoning**: Why this action is necessary
- **Description**: What element you're targeting
- **Execution**: The actual action to perform

## Available Functions

### 1. click
Click on elements by describing what you want to click.

**Parameters:**
- `description`: Description of the element to click (e.g., "search button", "login link", "submit button")
- `reasoning`: Reasoning for why this element should be clicked

### 2. type
Type text into input fields by describing the field.

**Parameters:**
- `text`: Text to type into the input field
- `field_description`: Description of the input field (e.g., "search box", "username field", "email input")
- `reasoning`: Reasoning for why this text should be typed in this field

### 3. scroll
Scroll the page in different directions.

**Parameters:**
- `direction`: Direction to scroll (up, down, left, right)
- `reasoning`: Reasoning for why scrolling in this direction is needed

### 4. wait
Wait for 2 seconds (default) to allow page to load or elements to appear.

**Parameters:**
- `reasoning`: Reasoning for why waiting is necessary

### 5. stop
Stop the current task and provide a final answer.

**Parameters:**
- `answer`: Final answer or result of the task
- `reasoning`: Reasoning for why the task is complete

## Usage

```python
from agent import construct_agent

# Create args
args = argparse.Namespace()
args.temperature = 0.0
args.top_p = 0.9
args.max_tokens = 10000
args.max_obs_length = 1920

# Construct agent
agent = construct_agent(args)

# Use the agent
trajectory = []
intent = "Click on the search button"
meta_data = {"action_history": ["None"]}

action = agent.next_action_custom(trajectory, intent, meta_data)
```

## vLLM Setup

To use this agent, you need to run a vLLM server with qwen2.5-instruct-VL:

```bash
# Install vLLM
pip install vllm

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Instruct-VL \
    --host 0.0.0.0 \
    --port 8000
```

## Configuration

The agent is configured to connect to the vLLM server at `http://localhost:8000/v1`. You can modify the configuration in the `_configure_llm` method:

```python
llm_config = {
    'model_type': 'qwenvl_oai',
    'model': 'qwen2.5-instruct-VL',
    'model_server': 'http://localhost:8000/v1',  # vLLM server
    'api_key': 'EMPTY',
    'generate_cfg': {
        'max_retries': 10,
        'fncall_prompt_type': 'qwen',
        'temperature': args.temperature,
        'top_p': args.top_p,
        'max_tokens': args.max_tokens,
        'max_obs_length': args.max_obs_length
    }
}
```

## Action Validation

Action validation is applied **after** every generated action (not as a function call):

- **Click actions**: Validates that a description is provided
- **Type actions**: Ensures both text and field description are provided
- **Scroll actions**: Validates direction is one of the allowed values
- **Wait actions**: Always valid (default 2 seconds)
- **Stop actions**: Always valid

## Integration with GUI Agent

This function call agent integrates seamlessly with the main GUI Agent framework:

1. **Trajectory Processing**: Can process browser trajectory data
2. **Image Analysis**: Can analyze screenshots for context
3. **Action History**: Uses action history for better decision making
4. **Error Handling**: Robust error handling with fallback to none actions
5. **ReAct Workflow**: Follows reasoning-then-acting paradigm

## Example Workflow

1. Agent receives current page state (screenshot + metadata)
2. Agent analyzes the task and current state (Think)
3. Agent provides reasoning for the next action (Reason)
4. Agent proposes an action using function calling (Act)
5. Action is validated automatically
6. Validated action is executed
7. Process repeats until task completion

## Key Changes from Previous Version

1. **Removed coordinates**: Click and type actions now use descriptions instead of coordinates
2. **Added reasoning**: All functions require reasoning parameter
3. **Default wait time**: Wait function uses 2 seconds by default
4. **Action validation**: Moved from function call to automatic post-generation validation
5. **ReAct paradigm**: Implemented reasoning-before-acting approach
6. **Required answer**: Stop function now requires an answer parameter 