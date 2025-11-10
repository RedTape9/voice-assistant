"""
Unit tests for the calculator tool
"""
import sys
import os
# Add parent directory to path to import assistant module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from assistant import calculate


def test_calculator_basic_addition():
    """Test basic addition"""
    assert calculate("2 + 2") == "4"
    assert calculate("10 + 5") == "15"


def test_calculator_basic_subtraction():
    """Test basic subtraction"""
    assert calculate("10 - 5") == "5"
    assert calculate("100 - 50") == "50"


def test_calculator_basic_multiplication():
    """Test basic multiplication"""
    assert calculate("2 * 3") == "6"
    assert calculate("10 * 5") == "50"


def test_calculator_basic_division():
    """Test basic division"""
    assert calculate("10 / 2") == "5.0"
    assert calculate("100 / 4") == "25.0"


def test_calculator_modulo():
    """Test modulo operation"""
    assert calculate("10 % 3") == "1"
    assert calculate("20 % 7") == "6"


def test_calculator_power():
    """Test power operation"""
    assert calculate("2 ** 3") == "8"
    assert calculate("10 ** 2") == "100"


def test_calculator_negative_numbers():
    """Test negative numbers"""
    assert calculate("-5 + 3") == "-2"
    assert calculate("-10 * 2") == "-20"


def test_calculator_parentheses():
    """Test expressions with parentheses"""
    assert calculate("(2 + 3) * 4") == "20"
    assert calculate("2 * (3 + 4)") == "14"


def test_calculator_complex_expression():
    """Test complex mathematical expression"""
    assert calculate("(10 + 5) * 2 - 8 / 4") == "28.0"


def test_calculator_division_by_zero():
    """Test division by zero error handling"""
    result = calculate("10 / 0")
    assert "Error" in result
    assert "Division by zero" in result


def test_calculator_invalid_expression():
    """Test invalid expression handling"""
    result = calculate("2 +")
    assert "Error" in result


def test_calculator_unsupported_operation():
    """Test unsupported operations are rejected"""
    result = calculate("import os")
    assert "Error" in result


def test_calculator_function_calls():
    """Test that function calls are rejected"""
    result = calculate("eval('2+2')")
    assert "Error" in result


def test_calculator_no_variables():
    """Test that variables are not allowed"""
    result = calculate("x = 5")
    assert "Error" in result


if __name__ == "__main__":
    # Run tests manually
    test_functions = [
        test_calculator_basic_addition,
        test_calculator_basic_subtraction,
        test_calculator_basic_multiplication,
        test_calculator_basic_division,
        test_calculator_modulo,
        test_calculator_power,
        test_calculator_negative_numbers,
        test_calculator_parentheses,
        test_calculator_complex_expression,
        test_calculator_division_by_zero,
        test_calculator_invalid_expression,
        test_calculator_unsupported_operation,
        test_calculator_function_calls,
        test_calculator_no_variables,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: Unexpected error - {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
