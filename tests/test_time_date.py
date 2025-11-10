"""
Unit tests for time and date functions
"""
import sys
import os
import re
import datetime
# Add parent directory to path to import assistant module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from assistant import get_current_time, get_current_date


def test_get_current_time_format():
    """Test that get_current_time returns proper format"""
    time_str = get_current_time()
    # Should match format like "02:30 PM"
    pattern = r'^\d{2}:\d{2} (AM|PM)$'
    assert re.match(pattern, time_str), f"Time format incorrect: {time_str}"


def test_get_current_time_is_string():
    """Test that get_current_time returns a string"""
    time_str = get_current_time()
    assert isinstance(time_str, str)


def test_get_current_date_format():
    """Test that get_current_date returns proper format"""
    date_str = get_current_date()
    # Should match format like "January 01, 2024"
    pattern = r'^[A-Z][a-z]+ \d{2}, \d{4}$'
    assert re.match(pattern, date_str), f"Date format incorrect: {date_str}"


def test_get_current_date_is_string():
    """Test that get_current_date returns a string"""
    date_str = get_current_date()
    assert isinstance(date_str, str)


def test_get_current_date_matches_today():
    """Test that get_current_date returns today's date"""
    date_str = get_current_date()
    today = datetime.datetime.now()
    expected = today.strftime("%B %d, %Y")
    assert date_str == expected


if __name__ == "__main__":
    # Run tests manually
    test_functions = [
        test_get_current_time_format,
        test_get_current_time_is_string,
        test_get_current_date_format,
        test_get_current_date_is_string,
        test_get_current_date_matches_today,
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
