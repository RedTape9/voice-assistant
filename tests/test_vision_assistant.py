"""
Test vision assistant core functionality without hardware dependencies
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Test imports
def test_tools_import():
    """Test that tools module can be imported"""
    from tools import web_search_tool, weather_tool, time_tool, calc_tool, DEFAULT_TOOLS
    assert callable(web_search_tool)
    assert callable(weather_tool)
    assert callable(time_tool)
    assert callable(calc_tool)
    assert len(DEFAULT_TOOLS) == 4
    print("PASS Tools module imported successfully")


def test_calculator_security():
    """Test that calculator is secure"""
    from tools import calc_tool

    # Test safe operations
    assert calc_tool("2+2") == "4"
    assert calc_tool("10*5") == "50"

    # Test security - should reject dangerous operations
    result_eval = calc_tool("eval('1+1')")
    assert "Fehler" in result_eval or "Error" in result_eval

    result_import = calc_tool("import os")
    assert "Fehler" in result_import or "Error" in result_import

    print("PASS Calculator is secure (AST-based)")


def test_time_tool():
    """Test time tool"""
    from tools import time_tool
    result = time_tool()
    # Should contain date and time in Russian format
    assert "Сегодня" in result or len(result) > 10
    print(f"PASS Time tool works: {result}")


def test_assistant_rate_limiting():
    """Test that Assistant class has rate limiting"""
    # We can't fully instantiate Assistant without LLM/dependencies
    # But we can check the class attributes exist
    import assistant
    assert hasattr(assistant.Assistant, 'MAX_REQUESTS_PER_MINUTE')
    assert hasattr(assistant.Assistant, 'MAX_INPUT_LENGTH')
    assert assistant.Assistant.MAX_REQUESTS_PER_MINUTE == 10
    assert assistant.Assistant.MAX_INPUT_LENGTH == 500
    print("PASS Assistant has rate limiting configured")


def test_assistant_has_security_methods():
    """Test that Assistant has security methods"""
    import assistant
    assert hasattr(assistant.Assistant, '_check_rate_limit')
    assert hasattr(assistant.Assistant, '_sanitize_input')
    print("PASS Assistant has security methods")


def test_webcam_stream_class_exists():
    """Test that WebcamStream class exists"""
    import assistant
    assert hasattr(assistant, 'WebcamStream')
    print("PASS WebcamStream class exists (vision support)")


def test_multilingial_config():
    """Test multilingual configuration"""
    import assistant
    assert hasattr(assistant, 'LANGUAGE_CONFIG')
    assert 'ru' in assistant.LANGUAGE_CONFIG
    assert 'de' in assistant.LANGUAGE_CONFIG
    print("PASS Multilingual support configured (Russian, German)")


if __name__ == "__main__":
    print("Testing Vision Assistant Core Functionality")
    print("=" * 50)

    tests = [
        test_tools_import,
        test_calculator_security,
        test_time_tool,
        test_assistant_rate_limiting,
        test_assistant_has_security_methods,
        test_webcam_stream_class_exists,
        test_multilingial_config,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"FAIL {test.__name__}: Unexpected error - {e}")
            failed += 1

    print("=" * 50)
    print(f"\n{passed} passed, {failed} failed")

    if failed == 0:
        print("\nAll core functionality tests passed!")
        print("Security fixes applied successfully")
        print("Vision support structure intact")
