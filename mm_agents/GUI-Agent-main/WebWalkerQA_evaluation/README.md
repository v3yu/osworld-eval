# WebWalkerQA Evaluation

This module provides evaluation capabilities for the WebWalkerQA benchmark using the GUI Agent.

## Overview

WebWalkerQA is a web navigation and question-answering benchmark that tests an agent's ability to:
1. Navigate to a specific website (using `root_url`)
2. Find information on the website
3. Answer questions based on the found information

## Dataset Structure

The WebWalkerQA dataset contains:
- **question**: The question to be answered
- **answer**: The reference answer
- **root_url**: The starting URL for navigation
- **info**: Additional metadata (domain, source_website, etc.)

## Components

### 1. Evaluator (`evaluator.py`)

Three evaluator classes are provided:

#### `WebWalkerQAEvaluator`
- Uses LLM fuzzy matching to evaluate answer correctness
- Compares predicted answer with reference answer using the same LLM approach as MMInA
- Returns a score between 0.0 and 1.0

#### `WebWalkerQAURLEvaluator`
- Checks if the final URL is within the correct domain
- Verifies that the agent navigated to the right website
- Returns 1.0 if final URL contains the root URL, 0.0 otherwise

#### `WebWalkerQACombinedEvaluator`
- Combines both answer and URL evaluation
- Both answer and URL must be correct for a perfect score
- Returns the product of answer_score × url_score

### 2. Test Runner (`test_runner.py`)

The `WebWalkerQATestRunner` class provides:

- **Data Loading**: Loads WebWalkerQA data from HuggingFace dataset
- **Task Execution**: Runs individual WebWalkerQA tasks
- **Evaluation**: Evaluates results using the specified evaluator
- **Results Tracking**: Saves detailed results and metrics

### 3. Helper Functions (`helper_functions.py`)

Utility functions for:
- Cleaning answers and URLs
- Extracting data from WebWalkerQA configs
- Creating task configurations

## Usage

### Basic Usage

```python
from WebWalkerQA_evaluation import WebWalkerQATestRunner, WebWalkerQAEvaluator
from agent.agent import FunctionCallAgent
from browser_env import BrowserEnv

# Initialize components
agent = FunctionCallAgent(...)
env = BrowserEnv(...)
evaluator = WebWalkerQAEvaluator()

# Create test runner
runner = WebWalkerQATestRunner(
    agent=agent,
    env=env,
    evaluator=evaluator,
    output_dir="webwalkerqa_results",
    max_steps=50,
    timeout=300
)

# Run evaluation
results = runner.run_evaluation(
    split="silver",  # or "main"
    max_samples=10   # limit number of samples
)
```

### Command Line Usage

```bash
cd GUI-Agent/WebWalkerQA_evaluation

# Run evaluation with default settings
python test_runner.py --split silver --max_samples 10

# Run with custom settings
python test_runner.py \
    --split main \
    --max_samples 50 \
    --output_dir my_results \
    --max_steps 100 \
    --timeout 600
```

## Configuration

### Dataset Splits

- **main**: Main evaluation split
- **silver**: Silver evaluation split (recommended for testing)

### Evaluation Parameters

- **max_steps**: Maximum number of steps per task (default: 50)
- **timeout**: Timeout per task in seconds (default: 300)
- **headless**: Run browser in headless mode (default: True)

## Output

The evaluation produces:

1. **Individual Task Results**: Each task saves its config and results
2. **Overall Results**: `results.json` with all results and metrics
3. **Metrics**: Average score, success rate, step statistics

### Results Structure

```json
{
  "results": [
    {
      "task_id": "webwalkerqa_conference_0",
      "question": "Which program funded the SUPPLY...",
      "root_url": "https://ehaweb.org/",
      "reference_answer": "The SUPPLY project that EHA is...",
      "final_answer": "The SUPPLY project...",
      "score": 1.0,
      "steps": 15,
      "final_url": "https://ehaweb.org/organization/newsroom/...",
      "trajectory": [...],
      "config": {...}
    }
  ],
  "metrics": {
    "total_tasks": 10,
    "average_score": 0.85,
    "success_rate": 0.8,
    "average_steps": 12.3,
    "score_distribution": {
      "perfect": 6,
      "partial": 2,
      "failed": 2
    }
  }
}
```

## Integration with Training Data Collection

The WebWalkerQA evaluation can be integrated with the training data collection system:

```python
# Enable training data collection
from utils.training_data_collector import TrainingDataCollector

collector = TrainingDataCollector(
    output_dir="webwalkerqa_training_data",
    enabled=True
)

# The agent will automatically collect conversation data during evaluation
```

## Requirements

- pandas
- openai (for vLLM client)
- playwright
- browser_env
- agent framework

## Notes

1. **LLM Evaluation**: The evaluator uses a local vLLM server with Qwen2.5-VL-Instruct model
2. **URL Matching**: URL evaluation allows navigation within the same domain
3. **Error Handling**: Robust error handling for network issues and timeouts
4. **Memory Efficiency**: Results are saved incrementally to handle large evaluations

## Example

```python
# Load a sample from WebWalkerQA
sample = {
    "question": "Which program funded the SUPPLY project?",
    "answer": "The SUPPLY project that EHA is implementing is funded by...",
    "root_url": "https://ehaweb.org/",
    "info": {"domain": "conference", "source_website": ["https://ehaweb.org/..."]}
}

# Create config
config = create_webwalkerqa_config(
    question=sample["question"],
    answer=sample["answer"], 
    root_url=sample["root_url"],
    info=sample["info"]
)

# Run task
result = runner.run_single_task(config)
print(f"Score: {result['score']:.3f}")
``` 