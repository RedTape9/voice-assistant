"""
LangChain agent with various tools for the voice assistant
"""
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
import datetime
from config import MODEL_NAME, MODEL_TEMPERATURE


def get_current_time():
    """Get the current time"""
    return datetime.datetime.now().strftime("%I:%M %p")


def get_current_date():
    """Get the current date"""
    return datetime.datetime.now().strftime("%B %d, %Y")


def calculate(expression):
    """
    Safely evaluate a mathematical expression
    
    Args:
        expression (str): Mathematical expression to evaluate
        
    Returns:
        str: Result of the calculation
    """
    try:
        # Only allow safe mathematical operations
        allowed_chars = set('0123456789+-*/().% ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


class Assistant:
    """LangChain-based assistant with various tools"""
    
    def __init__(self, api_key):
        """
        Initialize the assistant
        
        Args:
            api_key (str): OpenAI API key
        """
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=MODEL_TEMPERATURE,
            api_key=api_key
        )
        
        # Create tools
        self.tools = self._create_tools()
        
        # Create the agent using LangGraph
        self.agent = create_react_agent(self.llm, self.tools)
    
    def _create_tools(self):
        """Create the tools for the assistant"""
        tools = []
        
        # Wikipedia tool
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        tools.append(wikipedia)
        
        # Search tool
        search = DuckDuckGoSearchRun()
        tools.append(search)
        
        # Time tool
        tools.append(
            StructuredTool.from_function(
                func=get_current_time,
                name="CurrentTime",
                description="Useful for getting the current time. No input needed."
            )
        )
        
        # Date tool
        tools.append(
            StructuredTool.from_function(
                func=get_current_date,
                name="CurrentDate",
                description="Useful for getting the current date. No input needed."
            )
        )
        
        # Calculator tool
        tools.append(
            StructuredTool.from_function(
                func=calculate,
                name="Calculator",
                description="Useful for performing mathematical calculations. Input should be a mathematical expression like '2+2' or '10*5'."
            )
        )
        
        return tools
    
    def process(self, user_input):
        """
        Process user input and generate a response
        
        Args:
            user_input (str): The user's input text
            
        Returns:
            str: The assistant's response
        """
        try:
            # Invoke the agent with the user input
            result = self.agent.invoke({
                "messages": [("user", user_input)]
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
