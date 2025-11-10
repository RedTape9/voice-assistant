# Contributing to Voice Assistant

Thank you for considering contributing to the Voice Assistant project! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Keep discussions professional and on-topic

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git for version control
- Virtual environment tool (venv, virtualenv, or conda)
- Microphone for testing voice features
- OpenAI API key for testing

### Setting Up Development Environment

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/voice-assistant.git
   cd voice-assistant
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # Windows Git Bash
   source .venv/bin/activate      # Linux/macOS
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

5. **Run tests to verify setup**
   ```bash
   python tests/test_calculator.py
   python tests/test_time_date.py
   ```

## Project Structure

```
voice-assistant/
├── assistant.py         # Main LangChain agent and tools
├── voice_input.py       # Speech recognition handler
├── voice_output.py      # Text-to-speech handler
├── config.py            # Configuration settings
├── main.py              # Application entry point
├── tests/               # Unit tests
│   ├── test_calculator.py
│   └── test_time_date.py
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
├── README.md            # User documentation
└── CONTRIBUTING.md      # This file
```

## Development Guidelines

### Code Style

- **PEP 8**: Follow Python PEP 8 style guidelines
- **Type Hints**: Use type hints for all function signatures
- **Docstrings**: Include docstrings for all modules, classes, and functions
- **Line Length**: Maximum 100 characters per line
- **Imports**: Group imports (standard library, third-party, local)

Example:
```python
def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression

    Args:
        expression (str): Mathematical expression to evaluate

    Returns:
        str: Result of the calculation
    """
    # Implementation here
```

### Naming Conventions

- **Classes**: PascalCase (e.g., `VoiceInput`, `Assistant`)
- **Functions/Methods**: snake_case (e.g., `get_current_time`, `process`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_REQUESTS_PER_MINUTE`)
- **Private methods**: Prefix with underscore (e.g., `_sanitize_input`)

### Security Best Practices

1. **Never commit secrets**: API keys, passwords, or tokens
2. **Input validation**: Always sanitize and validate user input
3. **Safe evaluation**: Never use `eval()` or `exec()` with user input
4. **Rate limiting**: Implement rate limiting for API calls
5. **Error handling**: Don't expose sensitive information in error messages

### Testing

#### Writing Tests

- Create test files in the `tests/` directory
- Prefix test files with `test_`
- Prefix test functions with `test_`
- Test both success and failure cases
- Include edge cases and boundary conditions

Example:
```python
def test_calculator_division_by_zero():
    """Test division by zero error handling"""
    result = calculate("10 / 0")
    assert "Error" in result
    assert "Division by zero" in result
```

#### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python tests/test_calculator.py

# Run with coverage
pytest --cov=. tests/
```

### Adding New Features

#### Adding a New Tool

1. **Create the tool function** in `assistant.py`:
   ```python
   def my_new_tool(input: str) -> str:
       """Tool description for the LLM"""
       # Implementation
       return result
   ```

2. **Add type hints and docstrings**

3. **Register the tool** in `_create_tools()`:
   ```python
   tools.append(
       StructuredTool.from_function(
           name="MyNewTool",
           func=my_new_tool,
           description="Description for the agent"
       )
   )
   ```

4. **Write tests** for the new tool in `tests/test_my_new_tool.py`

5. **Update documentation** in README.md

#### Adding Configuration Options

1. Add the setting to `config.py`
2. Add documentation in README.md
3. Add to `.env.example` if it's an environment variable
4. Provide sensible defaults

### Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes**
   - Write clean, documented code
   - Follow the coding style guidelines
   - Add tests for new functionality

3. **Test thoroughly**
   ```bash
   python tests/test_calculator.py
   python tests/test_time_date.py
   # Run your new tests
   ```

4. **Commit with clear messages**
   ```bash
   git add .
   git commit -m "Add feature: description of what you added"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/my-new-feature
   ```

6. **Create a Pull Request**
   - Use a clear, descriptive title
   - Describe what changes you made and why
   - Reference any related issues
   - Include screenshots/examples if applicable

### Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] All functions have type hints and docstrings
- [ ] Tests added/updated for changes
- [ ] All tests pass
- [ ] Documentation updated (README.md if needed)
- [ ] No secrets or sensitive data committed
- [ ] Commit messages are clear and descriptive

## Common Tasks

### Running the Application

```bash
python main.py
```

### Updating Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Finding Audio Devices

```bash
# List audio devices (if list_audio_devices.py exists)
python list_audio_devices.py
```

## Reporting Issues

When reporting bugs or requesting features:

1. **Search existing issues** first
2. **Use issue templates** if available
3. **Provide details**:
   - Python version
   - Operating system
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages/logs
   - Screenshots if applicable

## Questions?

If you have questions:
- Check the README.md
- Review existing issues and discussions
- Create a new issue with the "question" label

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

## Thank You!

Your contributions help make this project better for everyone. We appreciate your time and effort!
