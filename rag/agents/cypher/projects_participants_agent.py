from llama_index.core.agent.workflow import FunctionAgent

from rag.llm import get_llama_indexllm
from rag.tools.cypher.participant_information import get_participant_information
from rag.tools.cypher.project_information import get_project_info

projects_participants_agent = FunctionAgent(
    name="EuropeanProjectsExpertAgent",
    description="This agent provides information about european projects.",
    system_prompt=(
        """
        You are an expert providing information about european projects.
        Be as helpful as possible and return as much information as possible.
        Do not answer any questions that do not relate to projects, organisations, grants, fundings, or participants.

        Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

        Use the following tools:

        **get_participant_information** tool to get information about a participant. 

        **get_project_info** tool to get information about a project. 
        
        If you use both the tools, merge the answers and combine them in a final professional answer.

        """
    ),
    llm=get_llama_indexllm(),
    tools=[get_project_info, get_participant_information],
    can_handoff_to=["PotentialCollaboratorsAgent"],
)
