# GUI Agent

A simplified and modular GUI Agent framework based on MMInA, rewritten using the Qwen-Agent framework structure.

## Structure

```
GUI-Agent/
├── run.py                 # Main entry point - simplified
├── config/                # Configuration modules
│   ├── __init__.py
│   └── argument_parser.py # Command line argument parsing
├── agent/                 # Agent implementation
│   ├── __init__.py
│   ├── agent.py           # Function call agent with ReAct
│   └── README.md          # Agent documentation
├── actions/               # Action creation and validation
│   ├── __init__.py
│   ├── action_creator.py  # Action creation functions
│   └── README.md          # Actions documentation
├── evaluation/            # Evaluation and testing modules
│   ├── __init__.py
│   ├── evaluator.py       # Evaluation functionality
│   └── test_runner.py     # Main test execution logic
└── utils/                 # Utility modules
    ├── __init__.py
    ├── action_check.py    # Action self-check and retry
    ├── early_stop.py      # Early stopping logic
    ├── fallback_answer.py # Fallback answer generation
    ├── help_functions.py  # General utility functions
    └── logging_setup.py   # Logging configuration
```

## Main Components

### run.py
The main entry point that has been simplified to focus on the core execution flow:
- Setup logging and environment
- Load models and agent
- Prepare test files
- Run tests

### agent/agent.py
Contains the function call agent implementation using the Qwen-Agent framework with ReAct paradigm:
- Function calling with vLLM integration
- ReAct (Reasoning and Acting) paradigm
- Action validation and processing
- Multimodal support for GUI interactions

### actions/action_creator.py
Contains all action creation and validation functions:
- Action creation for click, type, scroll, wait, stop operations
- Action validation logic
- Consistent action structure across the application

### config/argument_parser.py
Contains all command line argument definitions, extracted from the original run.py for better organization.

### evaluation/test_runner.py
Contains the main test execution logic that was previously in the `test()` function in run.py. This class handles:
- Environment initialization
- Test file processing
- Trajectory management
- Result evaluation and saving

### utils/
Various utility modules that were extracted from the original run.py:
- **action_check.py**: Handles action validation and retry logic
- **early_stop.py**: Implements early stopping conditions
- **fallback_answer.py**: Generates fallback answers when agent stops early
- **help_functions.py**: General utility functions
- **logging_setup.py**: Logging configuration

## Usage

The simplified structure maintains the same functionality as the original but with better organization:

```bash
python run.py --domain multi567 --model qwen2.5-vl --result_dir ./results
```

## Benefits of the Simplified Structure

1. **Separation of Concerns**: Each module has a specific responsibility
2. **Maintainability**: Easier to modify individual components
3. **Testability**: Individual modules can be tested in isolation
4. **Readability**: The main run.py file is now much cleaner and easier to understand
5. **Modularity**: Components can be easily replaced or extended

## Migration Notes

This simplified version maintains compatibility with the original MMInA codebase while providing a cleaner, more maintainable structure. All the original functionality has been preserved but reorganized into logical modules. 