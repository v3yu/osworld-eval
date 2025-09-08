# Actions Module

This module contains all the action creation functions for the GUI Agent, providing a centralized location for creating and validating browser actions.

## Functions

### Action Creation Functions

#### `create_click_action(description: str, reasoning: str = "") -> Action`
Creates a click action based on element description.

**Parameters:**
- `description`: Description of the element to click (e.g., "search button", "login link")
- `reasoning`: Reasoning for why this element should be clicked

**Returns:**
- Action dictionary for clicking

#### `create_type_action(text: str, field_description: str, reasoning: str = "") -> Action`
Creates a type action based on field description.

**Parameters:**
- `text`: Text to type into the input field
- `field_description`: Description of the input field (e.g., "search box", "username field")
- `reasoning`: Reasoning for why this text should be typed in this field

**Returns:**
- Action dictionary for typing

#### `create_scroll_action(direction: str, reasoning: str = "") -> Action`
Creates a scroll action.

**Parameters:**
- `direction`: Direction to scroll (up, down, left, right)
- `reasoning`: Reasoning for why scrolling in this direction is needed

**Returns:**
- Action dictionary for scrolling

#### `create_wait_action(seconds: float = 2.0, reasoning: str = "") -> Action`
Creates a wait action with default 2 seconds.

**Parameters:**
- `seconds`: Number of seconds to wait (default: 2.0)
- `reasoning`: Reasoning for why waiting is necessary

**Returns:**
- Action dictionary for waiting

#### `create_stop_action(answer: str, reasoning: str = "") -> Action`
Creates a stop action with answer.

**Parameters:**
- `answer`: Final answer or result of the task
- `reasoning`: Reasoning for why the task is complete

**Returns:**
- Action dictionary for stopping

#### `create_none_action() -> Action`
Creates a none action (no action to take).

**Returns:**
- Action dictionary for no action

### Utility Functions

#### `create_action_from_function_call(func_name: str, func_args: Dict[str, Any]) -> Action`
Creates an action from function call parameters.

**Parameters:**
- `func_name`: Name of the function called
- `func_args`: Arguments passed to the function

**Returns:**
- Action dictionary

#### `validate_action(action: Action) -> bool`
Validates if an action is appropriate.

**Parameters:**
- `action`: Action to validate

**Returns:**
- True if action is valid, False otherwise

## Usage

```python
from actions import (
    create_click_action,
    create_type_action,
    create_scroll_action,
    create_wait_action,
    create_stop_action,
    create_none_action,
    validate_action
)

# Create a click action
click_action = create_click_action(
    description="search button",
    reasoning="Need to click the search button to submit the query"
)

# Create a type action
type_action = create_type_action(
    text="example query",
    field_description="search box",
    reasoning="Need to type the search query into the search box"
)

# Validate an action
is_valid = validate_action(click_action)
```

## Action Structure

All actions follow a consistent structure:

```python
{
    'action_type': ActionTypes.CLICK,  # or TYPE, SCROLL, etc.
    'description': 'element description',  # for click actions
    'text': 'text to type',  # for type actions
    'field_description': 'field description',  # for type actions
    'direction': 'up/down/left/right',  # for scroll actions
    'seconds': 2.0,  # for wait actions
    'reasoning': 'reasoning for the action'
}
```

## Validation Rules

- **Click actions**: Must have a non-empty description
- **Type actions**: Must have both text and field description
- **Scroll actions**: Direction must be one of ['up', 'down', 'left', 'right']
- **Wait actions**: Always valid (default 2 seconds)
- **Stop actions**: Always valid
- **None actions**: Always valid

## Integration

This module is used by:
- **Agent module**: For creating actions from function calls
- **Test runner**: For action validation and processing
- **Other modules**: For consistent action creation across the application 