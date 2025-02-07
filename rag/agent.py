from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

from rag.graph import graph_db

from rag.tools.sparql import project_information, participant_information
from rag import llm

"""
chat_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert in the EUropean Research Information, real-world entities such as European projects, organizations, "
     "and research results. You describe administrative details of research projects, such as funding, start and end dates, "
     "and project participants."),
    ("human", "{input}")
])
"""

tools = [
    Tool.from_function(
        name="Find project information",
        description="Provide a list of information about a project such as the project title, project URL, project abstract, and project status",
        func=project_information.sparql_qa,
        return_direct=True
    ),
    Tool.from_function(
        name="Find participants information",
        description="Provide a list of information about participants, such as which participants are involved in a project, their country, and the number of participants in a project",
        func=participant_information.sparql_qa,
        return_direct=True
    )
]

agent_prompt = PromptTemplate.from_template("""
You are an expert providing information about european projects.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to projects, organisations, grants, fundings, or participants.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tools, please use the following format:

```
Thought: Do I need to use a tools? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tools, you MUST use the format:

```
Thought: Do I need to use a tools? No
Final Answer: [your response here]
```

Begin!

New input: {input}

{agent_scratchpad}
""")

agent = create_react_agent(llm.get_llm(), tools, agent_prompt)

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
        response = agent_executor.invoke({
            "schema": graph_db.schema,
            "input": prompt
        })
        return response['output']['result']
    except Exception as e:
        print(e)
        return f"An error occured while processing the request. Please try again."
