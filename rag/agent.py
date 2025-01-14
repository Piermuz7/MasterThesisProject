from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from rag.llm import llm

# TODO
tools = []

# TODO
agent_prompt = PromptTemplate.from_template("""
""")

agent = create_react_agent(llm, tools, agent_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


def generate_response(prompt):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """
    try:
        response = agent_executor.invoke({"input": prompt})
        return response['output']
    except Exception as e:
        print(e)
        return f"An error occured while processing the request. Please try again."