"""
LangChain agent with various tools for the voice assistant
"""
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
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
        
        # Create the agent
        self.agent = self._create_agent()
    
    def _create_tools(self):
        """Create the tools for the assistant"""
        tools = []
        
        # Wikipedia tool
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        tools.append(
            Tool(
                name="Wikipedia",
                func=wikipedia.run,
                description="Useful for looking up factual information on Wikipedia. Input should be a search query."
            )
        )
        
        # Search tool
        search = DuckDuckGoSearchRun()
        tools.append(
            Tool(
                name="Search",
                func=search.run,
                description="Useful for searching the internet for current information. Input should be a search query."
            )
        )
        
        # Time tool
        tools.append(
            Tool(
                name="CurrentTime",
                func=get_current_time,
                description="Useful for getting the current time. No input needed."
            )
        )
        
        # Date tool
        tools.append(
            Tool(
                name="CurrentDate",
                func=get_current_date,
                description="Useful for getting the current date. No input needed."
            )
        )
        
        # Calculator tool
        tools.append(
            Tool(
                name="Calculator",
                func=calculate,
                description="Useful for performing mathematical calculations. Input should be a mathematical expression like '2+2' or '10*5'."
            )
        )
        
        return tools
    
    def _create_agent(self):
        """Create the LangChain agent"""
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful voice assistant. Be concise and friendly in your responses. "
                      "You have access to various tools to help answer questions. "
                      "Use them when needed to provide accurate information."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        
        # Create the agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        return agent_executor
    
    def process(self, user_input):
        """
        Process user input and generate a response
        
        Args:
            user_input (str): The user's input text
            
        Returns:
            str: The assistant's response
        """
        try:
            response = self.agent.invoke({"input": user_input})
            return response["output"]
        except Exception as e:
            return f"I encountered an error: {str(e)}"
