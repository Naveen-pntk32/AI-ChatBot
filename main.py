from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path(__file__).with_name(".env"))

def _build_model() -> ChatOpenAI:
    """Create ChatOpenAI configured for OpenAI or OpenRouter based on env vars."""
    # Prefer OpenRouter if key is present
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")

    if openrouter_key:
        base_url = base_url or "https://openrouter.ai/api/v1"
        api_key = openrouter_key

    model_name = (
        os.getenv("OPENAI_MODEL")
        or os.getenv("OPENROUTER_MODEL")
        or "gpt-4o-mini"
    )

    # ChatOpenAI accepts base_url/api_key/model
    return ChatOpenAI(temperature=0, base_url=base_url, api_key=api_key, model=model_name)

@tool
def calculator(a: float, b: float) -> str:
    """Perform basic arithmetic addition of two numbers."""
    print("Tool has been called.")
    return f"The sum of {a} and {b} is {a + b}"
    
@tool
def say_hello(name: str) -> str:
    """Return a friendly personalized greeting."""
    print("Tool has been called.")
    return "Hello Naveen! I hope you are well today. How can I assist you?"

def main():
    try:
        model = _build_model()
        
        tools = [calculator, say_hello]
        agent_executor = create_react_agent(model, tools)
        
        print("Welcome! I'm your AI assistant. Type 'quit' to exit.")
        print("You can ask me to perform calculations or chat with me.")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input == "quit":
                break
            
            # Direct greeting shortcut to ensure consistent personalized message
            if user_input.lower() in {"hello", "hi", "hey", "greet me", "good morning", "good evening"}:
                print("\nAssistant: ", end="")
                print("Hello Naveen! I hope you are well today. How can I assist you?")
                continue
            
            print("\nAssistant: ", end="")
            try:
                for chunk in agent_executor.stream(
                    {"messages": [HumanMessage(content=user_input)]}
                ):
                    if "agent" in chunk and "messages" in chunk["agent"]:
                        for message in chunk["agent"]["messages"]:
                            print(message.content, end="")
                print()
            except Exception as e:
                print(f"Error: {e}")
                print("Please check your OpenAI API key and billing status.")
                print("You can also try using a different model or API key.")
                break
                
    except Exception as e:
        print(f"Failed to initialize AI assistant: {e}")
        print("Please check your OpenAI API key and billing status.")
        print("Make sure you have a valid .env file with OPENAI_API_KEY set.")
        
if __name__ == "__main__":
    main()
                