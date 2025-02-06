from llama_index.core.agent.workflow import FunctionAgent

from rag.llm import llama_index_llm
from rag.tools.participant_information import get_participant_information
from rag.tools.project_information import get_project_info

projects_participants_agent = FunctionAgent(
    name="EuropeanProjectsExpertAgent",
    description="This agent provides information about european projects.",
    system_prompt=(
        """
        You are an expert providing information about european projects.
        Be as helpful as possible and return as much information as possible.
        Do not answer any questions that do not relate to projects, organisations, grants, fundings, or participants.

        Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

        Use the get_participant_information tool to get information about a participant.

        Use the get_project_info tool to get information about a project. 

        """
    ),
    llm=llama_index_llm,
    tools=[get_project_info, get_participant_information],
    can_handoff_to=["PotentialCollaboratorsAgent"],
)
