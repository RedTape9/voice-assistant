# Voice Assistant Tests

This directory contains unit tests for the voice assistant components.

## Running Tests

### With pytest (recommended)

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_calculator.py

# Run with verbose output
pytest -v tests/
```

### Without pytest

Each test file can be run directly:

```bash
python tests/test_calculator.py
python tests/test_time_date.py
```

## Test Coverage

- **test_calculator.py**: Tests for the calculator tool
  - Basic arithmetic operations (addition, subtraction, multiplication, division)
  - Advanced operations (modulo, power, negative numbers)
  - Complex expressions with parentheses
  - Error handling (division by zero, invalid syntax)
  - Security (rejecting unsafe operations, function calls, variables)

- **test_time_date.py**: Tests for time and date functions
  - Time format validation
  - Date format validation
  - Date accuracy

## Adding New Tests

1. Create a new test file in the `tests/` directory with the prefix `test_`
2. Import the necessary modules from the parent directory
3. Write test functions with the prefix `test_`
4. Use assertions to validate expected behavior
5. Add error handling tests for edge cases

Example:
```python
def test_my_feature():
    """Test description"""
    result = my_function()
    assert result == expected_value
```

## Test Dependencies

The tests use only Python's standard library and the assistant modules. No additional dependencies are required for basic testing.

For enhanced testing capabilities, install:
```bash
pip install pytest pytest-cov
```

Run with coverage:
```bash
pytest --cov=. --cov-report=html tests/
```
