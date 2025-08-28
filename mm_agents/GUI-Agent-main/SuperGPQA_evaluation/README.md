# SuperGPQA Evaluation System

This module provides evaluation capabilities for the SuperGPQA (Super Graduate Program Qualifying Assessment) dataset, which contains multiple-choice questions from various academic disciplines.

## Overview

SuperGPQA is a comprehensive dataset of graduate-level multiple-choice questions covering:
- **Engineering** (29.7% of questions)
- **Philosophy** 
- **Medicine**
- **And many other disciplines**

Each question includes:
- A question text
- Multiple choice options (typically 9-10 options)
- Correct answer text and letter
- Discipline, field, and subfield metadata
- Difficulty level
- Calculation requirement flag

## Dataset Structure

Based on the dataset analysis, each SuperGPQA entry contains:

```json
{
  "uuid": "a8390c754538493ba59055689b4482aa",
  "question": "The common-mode rejection ratio of the first stage amplification circuit...",
  "options": [
    "the absolute value of the difference in the common-mode rejection ratio of A1 and A2 themselves",
    "all of the above",
    "the average of A1 and A2's common-mode rejection ratios",
    // ... more options
  ],
  "answer": "The difference in the common-mode rejection ratio of A1 and A2 themselves",
  "answer_letter": "I",
  "discipline": "Engineering",
  "field": "Electronic Science and Technology", 
  "subfield": "Circuits and Systems",
  "difficulty": "middle",
  "is_calculation": false
}
```

## Installation

1. **Login to HuggingFace** (required to access the dataset):
```bash
huggingface-cli login
```

2. **Install dependencies**:
```bash
pip install pandas openai
```

## Usage

### 1. Test the System
```bash
cd GUI-Agent
python test_supergpqa_evaluation.py
```

### 2. Run Full Evaluation
```bash
cd GUI-Agent
python run_supergpqa_evaluation.py --max_samples 10
```

### 3. Command Line Options

The SuperGPQA evaluation uses the centralized argument parser from `config/argument_parser.py`. Here are the key SuperGPQA-specific options:

#### Basic SuperGPQA Options:
```bash
--evaluation_type supergpqa          # Set evaluation type to SuperGPQA
--max_samples 10                     # Maximum number of questions to test
--supergpqa_data_path "path/to/data" # Path to SuperGPQA dataset
--evaluator_type semantic            # Evaluator type: semantic, letter, or combined
--timeout 300                        # Timeout per question in seconds
```

#### Browser Environment Options:
```bash
--headless                          # Run browser in headless mode
--viewport_width 1280               # Browser viewport width
--viewport_height 720               # Browser viewport height
--slow_mo 0                         # Slow motion for browser
--save_trace_enabled                # Save browser traces
--render_screenshot                 # Render screenshots during evaluation
```

#### Agent Configuration:
```bash
--model "qwen2.5-vl"                # Model to use
--max_steps 50                      # Maximum steps per question
--action_check                      # Enable action checking
--parsing_failure_th 3              # Parsing failure threshold
--repeating_action_failure_th 3     # Repeating action failure threshold
```

#### Memory and Training:
```bash
--save_examples_memory              # Save examples to memory
--collect_training_data             # Collect training data
--training_data_dir "training_data/supergpqa"  # Training data directory
```

### 4. Example Commands

```bash
# Basic evaluation with 10 questions
python run_supergpqa_evaluation.py --max_samples 10

# Evaluation with memory saving and action checking
python run_supergpqa_evaluation.py --max_samples 10 --save_examples_memory --action_check

# Evaluation with custom model and evaluator
python run_supergpqa_evaluation.py --max_samples 10 --model "gpt-4" --evaluator_type combined

# Evaluation with custom data path
python run_supergpqa_evaluation.py --max_samples 10 --supergpqa_data_path "local/path/to/data.jsonl"

# Evaluation with planning and subtask decomposition
python run_supergpqa_evaluation.py --max_samples 10 --self_plan --subtask
```

### 5. Programmatic Usage

```python
import sys
sys.path.insert(0, '/path/to/GUI-Agent')

from config.argument_parser import config
from SuperGPQA_evaluation.test_runner import SuperGPQATestRunner
from agent.agent import FunctionCallAgent

# Get arguments from centralized parser
args = config()
args.evaluation_type = "supergpqa"
args.max_samples = 10

# Initialize agent
agent = FunctionCallAgent()

# Create test runner
runner = SuperGPQATestRunner(args, agent)

# Run evaluation
runner.run()
```

### Using Different Evaluators

#### 1. Semantic Evaluator (Default)
Evaluates answer correctness using LLM fuzzy matching:

```python
from SuperGPQA_evaluation import SuperGPQAEvaluator

evaluator = SuperGPQAEvaluator()
# Uses LLM to compare predicted vs correct answer
```

#### 2. Letter Evaluator
Checks if the predicted answer letter matches the correct letter:

```python
from SuperGPQA_evaluation import SuperGPQALetterEvaluator

evaluator = SuperGPQALetterEvaluator()
# Extracts A, B, C, D, etc. from prediction and compares
```

#### 3. Combined Evaluator
Requires both semantic correctness AND correct letter:

```python
from SuperGPQA_evaluation import SuperGPQACombinedEvaluator

evaluator = SuperGPQACombinedEvaluator()
# Both semantic and letter must be correct
```

### Data Loading

```python
from SuperGPQA_evaluation import load_supergpqa_data

# Load from HuggingFace (default)
df = load_supergpqa_data()

# Load from local file
df = load_supergpqa_data("path/to/local/supergpqa.jsonl")

print(f"Loaded {len(df)} questions")
print(f"Disciplines: {df['discipline'].unique()}")
```

### Creating Test Configs

```python
from SuperGPQA_evaluation import create_supergpqa_config, format_question_with_options

# Create config from dataset row
config = create_supergpqa_config(
    question="What is the capital of France?",
    options=["London", "Berlin", "Paris", "Madrid"],
    answer="Paris",
    answer_letter="C",
    discipline="Geography",
    difficulty="easy"
)

# Format question for display
formatted = format_question_with_options(config["question"], config["options"])
print(formatted)
```

## Evaluation Process

1. **Data Loading**: Loads SuperGPQA dataset from HuggingFace
2. **Validation**: Filters valid questions with required fields
3. **Config Creation**: Creates test configurations with prompts
4. **Agent Execution**: Runs the agent on each question
5. **Evaluation**: Compares agent predictions with correct answers
6. **Results**: Saves detailed results and summary statistics

## Output Format

Results are saved in two formats:

### JSON Results (`supergpqa_results.json`)
```json
[
  {
    "test_id": 1,
    "uuid": "a8390c754538493ba59055689b4482aa",
    "question": "The common-mode rejection ratio...",
    "options": ["option1", "option2", ...],
    "correct_answer": "The difference in the common-mode rejection ratio...",
    "correct_letter": "I",
    "predicted_answer": "The difference in the common-mode rejection ratio...",
    "score": 1.0,
    "execution_time": 45.2,
    "discipline": "Engineering",
    "field": "Electronic Science and Technology",
    "subfield": "Circuits and Systems",
    "difficulty": "middle",
    "is_calculation": false,
    "trajectory_length": 15
  }
]
```

### CSV Results (`supergpqa_results.csv`)
Same data in CSV format for easy analysis in Excel/Python.

## Configuration

### Test Runner Configuration

```python
runner = SuperGPQATestRunner(
    agent=agent,
    browser_env=browser_env,
    evaluator=evaluator,
    output_dir="results/supergpqa",  # Output directory
    max_tests=10,                    # Maximum number of tests
    start_url="https://www.bing.com" # Starting URL for browser
)
```

### Evaluator Configuration

```python
evaluator = SuperGPQAEvaluator(
    vllm_client=None,           # vLLM client (optional)
    eval_tag="supergpqa"        # Evaluation tag for logging
)
```

## Example Test Script

```python
#!/usr/bin/env python3
"""Test script for SuperGPQA evaluation"""

import sys
sys.path.insert(0, '/lustre/scratch/users/guangyi.liu/agent/GUI-Agent')

from SuperGPQA_evaluation import SuperGPQAEvaluator, SuperGPQATestRunner
from SuperGPQA_evaluation.helper_functions import load_supergpqa_data, create_supergpqa_config
from browser_env import BrowserEnv
from agent.agent import FunctionCallAgent

def main():
    # Test data loading
    print("Loading SuperGPQA data...")
    df = load_supergpqa_data()
    
    if df.empty:
        print("Failed to load data. Make sure you're logged in with `huggingface-cli login`")
        return
    
    print(f"Loaded {len(df)} questions")
    print(f"Sample question: {df.iloc[0]['question'][:100]}...")
    
    # Test config creation
    print("\nTesting config creation...")
    sample_row = df.iloc[0]
    config = create_supergpqa_config(
        question=sample_row["question"],
        options=sample_row["options"],
        answer=sample_row["answer"],
        answer_letter=sample_row["answer_letter"],
        discipline=sample_row.get("discipline", ""),
        difficulty=sample_row.get("difficulty", "")
    )
    
    print(f"Created config with {len(config['options'])} options")
    
    # Test evaluator (if vLLM server is running)
    print("\nTesting evaluator...")
    try:
        evaluator = SuperGPQAEvaluator()
        
        # Mock trajectory
        mock_trajectory = [{"answer": sample_row["answer"]}]
        
        # Save config to file
        import json
        with open("test_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        score = evaluator(mock_trajectory, "test_config.json")
        print(f"Evaluation score: {score}")
        
    except Exception as e:
        print(f"Error testing evaluator: {e}")
        print("This is expected if vLLM server is not running")

if __name__ == "__main__":
    main()
```

## Requirements

- Python 3.8+
- pandas
- openai
- HuggingFace account (for dataset access)
- vLLM server (for LLM evaluation)

## Notes

- The dataset requires HuggingFace login: `huggingface-cli login`
- LLM evaluation requires a running vLLM server
- Questions start from Bing.com as specified
- Results include detailed metadata for analysis by discipline, difficulty, etc. 