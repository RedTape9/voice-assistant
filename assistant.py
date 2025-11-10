"""
LangChain agent with various tools for the voice assistant
"""
from typing import List, Any
import time
from collections import deque
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool, BaseTool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
import datetime
import ast
import operator
from config import MODEL_NAME, MODEL_TEMPERATURE


def get_current_time() -> str:
    """Get the current time"""
    return datetime.datetime.now().strftime("%I:%M %p")


def get_current_date() -> str:
    """Get the current date"""
    return datetime.datetime.now().strftime("%B %d, %Y")


def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression using AST parsing

    Args:
        expression (str): Mathematical expression to evaluate

    Returns:
        str: Result of the calculation
    """
    # Define safe operations
    safe_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def _eval_node(node):
        """Recursively evaluate AST nodes"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            op_type = type(node.op)
            if op_type not in safe_operators:
                raise ValueError(f"Unsupported operation: {op_type.__name__}")
            return safe_operators[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = _eval_node(node.operand)
            op_type = type(node.op)
            if op_type not in safe_operators:
                raise ValueError(f"Unsupported operation: {op_type.__name__}")
            return safe_operators[op_type](operand)
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')
        # Evaluate the AST safely
        result = _eval_node(tree.body)
        return str(result)
    except SyntaxError:
        return "Error: Invalid mathematical expression syntax"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error: Unable to calculate - {str(e)}"


class Assistant:
    """LangChain-based assistant with various tools"""

    # Rate limiting settings
    MAX_REQUESTS_PER_MINUTE = 10
    MAX_INPUT_LENGTH = 500

    def __init__(self, api_key: str) -> None:
        """
        Initialize the assistant

        Args:
            api_key (str): OpenAI API key
        """
        # Initialize the LLM
        self.llm: ChatOpenAI = ChatOpenAI(
            model=MODEL_NAME,
            temperature=MODEL_TEMPERATURE,
            api_key=api_key
        )

        # Create tools
        self.tools: List[BaseTool] = self._create_tools()

        # Create the agent using LangGraph
        self.agent: Any = create_react_agent(self.llm, self.tools)

        # Rate limiting: track request timestamps
        self.request_times: deque = deque(maxlen=self.MAX_REQUESTS_PER_MINUTE)

    def _create_tools(self) -> List[BaseTool]:
        """Create the tools for the assistant"""
        tools: List[BaseTool] = []
        
        # Wikipedia tool
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        tools.append(wikipedia)
        
        # Search tool
        search = DuckDuckGoSearchRun()
        tools.append(search)
        
        # Time tool
        tools.append(
            StructuredTool.from_function(
                name="CurrentTime",
                func=get_current_time,
                description="Useful for getting the current time. No input needed."
            )
        )
        
        # Date tool
        tools.append(
            StructuredTool.from_function(
                name="CurrentDate",
                func=get_current_date,
                description="Useful for getting the current date. No input needed."
            )
        )
        
        # Calculator tool
        tools.append(
            StructuredTool.from_function(
                name="Calculator",
                func=calculate,
                description="Useful for performing mathematical calculations. Input should be a mathematical expression like '2+2' or '10*5'."
            )
        )
        
        return tools
    
    def _check_rate_limit(self) -> bool:
        """
        Check if request is within rate limit

        Returns:
            bool: True if within rate limit, False otherwise
        """
        current_time = time.time()
        # Remove requests older than 60 seconds
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()

        # Check if we've exceeded the rate limit
        if len(self.request_times) >= self.MAX_REQUESTS_PER_MINUTE:
            return False

        # Add current request
        self.request_times.append(current_time)
        return True

    def _sanitize_input(self, user_input: str) -> str:
        """
        Sanitize user input

        Args:
            user_input (str): Raw user input

        Returns:
            str: Sanitized input

        Raises:
            ValueError: If input is invalid
        """
        # Strip whitespace
        sanitized = user_input.strip()

        # Check for empty input
        if not sanitized:
            raise ValueError("Input cannot be empty")

        # Check length
        if len(sanitized) > self.MAX_INPUT_LENGTH:
            raise ValueError(
                f"Input too long. Maximum {self.MAX_INPUT_LENGTH} characters allowed."
            )

        # Remove any control characters except newlines and tabs
        sanitized = ''.join(
            char for char in sanitized
            if char.isprintable() or char in '\n\t'
        )

        return sanitized

    def process(self, user_input: str) -> str:
        """
        Process user input and generate a response with rate limiting and sanitization

        Args:
            user_input (str): The user's input text

        Returns:
            str: The assistant's response
        """
        # Check rate limit
        if not self._check_rate_limit():
            return "Please slow down. You're making requests too quickly. Try again in a moment."

        # Sanitize input
        try:
            sanitized_input = self._sanitize_input(user_input)
        except ValueError as e:
            return f"Invalid input: {str(e)}"

        try:
            # Invoke the agent with the sanitized input
            result = self.agent.invoke({
                "messages": [("user", sanitized_input)]
            })

            # Extract the last message from the agent's response
            if "messages" in result and len(result["messages"]) > 0:
                last_message = result["messages"][-1]
                # Get the content from the message
                if hasattr(last_message, 'content'):
                    return last_message.content
                elif isinstance(last_message, dict) and 'content' in last_message:
                    return last_message['content']

            return "I'm sorry, I couldn't generate a response."
        except Exception as e:
            return f"I encountered an error: {str(e)}"
