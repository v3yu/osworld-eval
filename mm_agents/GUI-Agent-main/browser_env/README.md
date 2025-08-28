# Simplified Browser Environment

This module provides a simplified browser environment that supports both image and text observations.

## Overview

The browser environment has been simplified to focus on image observations while keeping the text observation structure for future extensions. Complex HTML parsing and accessibility tree processing have been removed, but the text observation framework remains for adding self-reflection, history descriptions, etc. later.

## Key Changes

### `envs.py`
- **Removed**: Complex HTML and accessibility tree processing
- **Removed**: Text observation types and processors
- **Simplified**: Only supports image observations
- **Streamlined**: Direct screenshot capture without complex parsing
- **Updated**: Step function signature to remove unnecessary parameters

### `processors.py`
- **Removed**: Complex HTML parsing and accessibility tree processing
- **Removed**: Complex caption and image merging functionality
- **Simplified**: `SimpleImageObservationProcessor` for screenshots
- **Added**: `SimpleTextObservationProcessor` that returns empty text (placeholder)
- **Kept**: Interaction point visualization (red circles on screenshots)

## Components

### ScriptBrowserEnv
The main browser environment class that:
- Manages Playwright browser instance
- Handles page navigation and actions
- Captures screenshots as observations
- Supports both regular and pixel-based actions

### SimpleImageObservationProcessor
A simplified observation processor that:
- Captures page screenshots
- Converts them to numpy arrays
- Optionally draws interaction points (red circles)
- Handles page loading waits

### SimpleTextObservationProcessor
A placeholder text processor that:
- Returns empty text for now
- Ready for future extensions (self-reflection, history, etc.)
- Maintains the text observation structure

## Usage

```python
from browser_env import ScriptBrowserEnv

# Create environment
env = ScriptBrowserEnv(
    headless=True,
    viewport_size={"width": 1280, "height": 720},
    sleep_after_execution=0.5
)

# Reset with config
observation, info = env.reset(options={"config_file": "task_config.json"})

# Execute actions
observation, reward, done, truncated, info = env.step(action, pixel_action=True)

# Close environment
env.close()
```

## Configuration

The environment can be configured with a JSON file containing:
```json
{
    "url": "https://example.com"
}
```

## Observation Format

Observations are returned as:
```python
{
    "image": numpy.ndarray,  # RGB screenshot as numpy array
    "text": ""               # Empty text for now, ready for future extensions
}
```

## Action Support

The environment supports:
- **Regular actions**: Standard browser actions (click, type, scroll, etc.)
- **Pixel actions**: Grounding model-based actions with coordinate detection

## Benefits

1. **Performance**: Faster execution without complex HTML parsing
2. **Simplicity**: Easier to understand and maintain
3. **Focus**: Concentrates on image-based interactions
4. **Extensibility**: Easy to add custom text processing later
5. **Compatibility**: Works with existing action parsers

## Future Extensions

When you're ready to add text processing (self-reflection, history description, etc.), you can:
1. Modify the `SimpleTextObservationProcessor.process()` method
2. Add your text processing logic (self-reflection, history, etc.)
3. The observation format already supports both image and text
4. The agent can already handle the combined observations 