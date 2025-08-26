# Training Data Collection System

This document describes the training data collection system for the GUI Agent, which captures LLM interactions for future training and fine-tuning.

## Features

- **Single Interactions**: Capture individual prompt-response pairs
- **Trajectories**: Organize multi-round interactions into logical sequences
- **Conversations**: Task-based conversation history with multiple rounds in one JSON
- **Actual Model Input**: Capture the exact input sent to the model (including function prompts)
- **Image Preservation**: Automatically convert images to base64 format for complete data preservation
- **Session Management**: Track and summarize data collection sessions

## Usage

### Basic Usage

```python
from utils.training_data_collector import TrainingDataCollector

# Create collector
collector = TrainingDataCollector(output_dir="training_data", enabled=True)

# Collect a single interaction
collector.collect_interaction(
    messages=[{"role": "user", "content": "Hello"}],
    response=[{"role": "assistant", "content": "Hi there!"}]
)
```

### Conversation History

For multi-round conversations about the same task:

```python
# Start a conversation for a specific task
collector.start_conversation(
    conversation_id="task_001",
    task_description="Navigate to settings and change theme"
)

# Add rounds to the conversation
collector.add_conversation_round(
    messages=[{"role": "user", "content": "Go to settings"}],
    response=[{"role": "assistant", "content": "I'll navigate to settings"}],
    actual_model_input=actual_input,  # Optional: actual model input
    functions=available_functions,     # Optional: available functions
    round_info={"action": "navigation"}
)

collector.add_conversation_round(
    messages=[{"role": "user", "content": "Change to dark theme"}],
    response=[{"role": "assistant", "content": "I'll change the theme"}],
    round_info={"action": "theme_change"}
)

# End the conversation and save to JSON
collector.end_conversation({
    "task_completed": True,
    "final_status": "success"
})
```

### Image Data Preservation

The system automatically preserves image data in messages:

```python
# Messages with images are automatically processed
messages_with_image = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this screenshot?"},
            {
                "type": "image_url",
                "image_url": {"url": "/path/to/screenshot.png"}
            }
        ]
    }
]

# Images are automatically converted to base64 data URLs
collector.collect_interaction(messages_with_image, response)
```

### Trajectories (Legacy)

For organizing multi-round interactions:

```python
# Start a trajectory
collector.start_trajectory("trajectory_001")

# Add rounds
collector.add_round(messages, response, round_info)

# End trajectory
collector.end_trajectory(trajectory_summary)
```

### Actual Model Input Collection

Capture the exact input sent to the model (including function prompts):

```python
# Collect actual model input
collector.collect_actual_model_input(
    original_messages=original_messages,
    actual_model_input=preprocessed_messages,
    functions=available_functions
)

# Or collect complete interaction with actual input
collector.collect_interaction_with_actual_input(
    original_messages=original_messages,
    actual_model_input=preprocessed_messages,
    response=model_response,
    functions=available_functions
)
```

## Data Structure

### Conversation History Format

```json
{
  "session_id": "uuid",
  "session_start": "2024-01-01T10:00:00",
  "conversation_id": "task_001",
  "conversation_start": "2024-01-01T10:00:00",
  "conversation_end": "2024-01-01T10:05:00",
  "task_description": "Navigate to settings and change theme",
  "total_rounds": 3,
  "rounds": [
    {
      "round_number": 1,
      "timestamp": "2024-01-01T10:00:00",
      "messages": [
        {
          "role": "user",
          "content": [
            {"type": "text", "text": "Go to settings"},
            {
              "type": "image_url",
              "image_url": {
                "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
              }
            }
          ]
        }
      ],
      "response": [{"role": "assistant", "content": "I'll navigate to settings"}],
      "actual_model_input": [...],
      "functions": [...],
      "round_info": {"action": "navigation"}
    }
  ],
  "conversation_summary": {
    "task_completed": true,
    "final_status": "success"
  },
  "metadata": {
    "filename": "conversation_20240101_100000_12345678_task_001.json",
    "filepath": "/path/to/file.json",
    "data_type": "conversation_history"
  }
}
```

### Single Interaction Format

```json
{
  "session_id": "uuid",
  "session_start": "2024-01-01T10:00:00",
  "timestamp": "2024-01-01T10:00:00",
  "messages": [
    {
      "role": "user",
      "content": "Hello"
    }
  ],
  "response": [
    {
      "role": "assistant",
      "content": "Hi there!"
    }
  ],
  "context": {},
  "metadata": {
    "filename": "interaction_20240101_100000_12345678.json",
    "filepath": "/path/to/file.json"
  }
}
```

### Actual Model Input Format

```json
{
  "session_id": "uuid",
  "session_start": "2024-01-01T10:00:00",
  "timestamp": "2024-01-01T10:00:00",
  "original_messages": [...],
  "actual_model_input": [
    {
      "role": "system",
      "content": "You are a helpful assistant. Available functions: [function descriptions...]"
    },
    {
      "role": "user",
      "content": "Hello"
    }
  ],
  "functions": [...],
  "context": {},
  "metadata": {
    "filename": "model_input_20240101_100000_12345678.json",
    "filepath": "/path/to/file.json",
    "input_type": "actual_model_input"
  }
}
```

## File Naming Convention

- **Single Interactions**: `interaction_{timestamp}_{uuid}.json`
- **Conversations**: `conversation_{timestamp}_{conversation_id}.json`
- **Trajectories**: `trajectory_{timestamp}_{trajectory_id}.json`
- **Model Inputs**: `model_input_{timestamp}_{uuid}.json`
- **Complete Interactions**: `complete_interaction_{timestamp}_{uuid}.json`

## Integration Points

### Agent Integration

The training data collector is integrated into the agent through the `LLMWrapper`:

```python
# In agent.py
if self.training_collector:
    wrapped_llm = wrap_llm(self.llm)
    self.llm = wrapped_llm
```

### Command Line Arguments

```bash
# Enable training data collection
python main.py --collect_training_data --training_data_dir my_data

# Disable training data collection
python main.py --no-collect_training_data
```

### Manual Usage

```python
# Start a conversation for a task
collector = get_collector()
collector.start_conversation("task_001", "Complete the form")

# The LLM wrapper will automatically add rounds to the conversation
# When the task is complete, end the conversation
collector.end_conversation({"status": "completed"})
```

## Session Summary

Get a summary of collected data:

```python
summary = collector.get_session_summary()
print(summary)
# Output:
# {
#   "session_id": "uuid",
#   "session_start": "2024-01-01T10:00:00",
#   "session_duration": "0:05:00",
#   "total_interactions": 5,
#   "total_trajectories": 2,
#   "total_conversations": 3,
#   "total_model_inputs": 10,
#   "total_complete_interactions": 8,
#   "total_trajectory_rounds": 6,
#   "total_conversation_rounds": 12,
#   "output_directory": "/path/to/training_data"
# }
```

## Image Processing

The system automatically processes images in messages:

1. **File Path Detection**: Detects image file paths in message content
2. **Base64 Conversion**: Converts image files to base64 data URLs
3. **MIME Type Detection**: Automatically determines correct MIME type
4. **Error Handling**: Gracefully handles missing or corrupted image files

Supported image formats:
- PNG (.png)
- JPEG (.jpg, .jpeg)
- GIF (.gif)
- BMP (.bmp)
- WebP (.webp)

## Best Practices

1. **Use Conversations for Tasks**: Start a conversation for each user task
2. **Include Task Descriptions**: Provide clear task descriptions for better organization
3. **Add Round Information**: Include relevant metadata in round_info
4. **End Conversations**: Always end conversations to save the data
5. **Monitor File Sizes**: Large images can create large JSON files
6. **Regular Cleanup**: Archive old training data periodically

## Testing

Run the test script to verify functionality:

```bash
cd GUI-Agent
python test_conversation_and_images.py
```

This will test:
- Conversation history collection
- Image data preservation
- Multiple conversations
- Session summaries 