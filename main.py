import os
import streamlit as st
from dotenv import load_dotenv
import io
import contextlib

# LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
# NEW - This is the correct import path now
from langchain.agents import create_tool_calling_agent


def setup_agent():
    """Initializes and returns the LangChain agent and executor."""
    load_dotenv()
    
    # Check for API key
    if "GOOGLE_API_KEY" not in os.environ:
        st.error("Error: GOOGLE_API_KEY not found. Please add it to your .env file.")
        st.stop()
        
    # 1. Set up the LLM (the "brain" of the agent)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

    # 2. Define the Tools (the "hands" of the agent)
    search_tool = DuckDuckGoSearchRun()
    tools = [search_tool] 

    # 3. Create the Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert researcher. Provide a clear, concise, and factual summary of the topic. You have access to a web search tool."),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # 4. Create the Agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 5. Create the AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor

def run_research_agent(agent_executor, question):
    """Invokes the agent with a question and captures its verbose output."""
    # Create a string buffer to capture the output
    string_io = io.StringIO()

    # Redirect stdout to the buffer
    with contextlib.redirect_stdout(string_io):
        try:
            response = agent_executor.invoke({"input": question})
        except Exception as e:
            # Handle potential errors during agent execution
            return f"An error occurred: {e}", ""

    # Get the content from the buffer
    verbose_output = string_io.getvalue()

    return response.get("output", "No output found."), verbose_output

# --- Streamlit Web Interface ---

st.set_page_config(page_title="AI Research Agent", layout="wide")
st.title("🤖 AI Topic Research Agent")

st.markdown("""
This agent uses Google's Gemini Pro model and LangChain to research a given topic by searching the web. 
Enter your question below to start.
""")

# Initialize agent
agent_executor = setup_agent()

# User input
user_question = st.text_input("Enter your research question:", placeholder="What is agentic AI and how does it relate to IT Operations (ITOps)?")

if st.button("Run Research"):
    if user_question:
        with st.spinner("The agent is thinking and researching..."):
            final_answer, verbose_log = run_research_agent(agent_executor, user_question)

            st.subheader("Final Answer:")
            st.markdown(final_answer)

            # Show the agent's thought process in an expander
            with st.expander("Show Agent's Thought Process"):
                st.text(verbose_log)
    else:
        st.warning("Please enter a research question.")