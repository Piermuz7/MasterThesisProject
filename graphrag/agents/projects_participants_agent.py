from SPARQLWrapper import SPARQLWrapper, JSON
from llama_index.core.agent.workflow import FunctionAgent

from graphrag import llm
from graphrag.tools.participant_information import get_participant_information
from graphrag.tools.project_information import get_project_info
from graphrag.embeddings.graph_embedding_service import GraphEmbeddingStore
import streamlit as st

embedding_store = GraphEmbeddingStore()

async def get_similar_projects(user_question: str) -> list[str]:
    "Finds similar project to a given project description"
    project_IRIs_by_similarity = embedding_store.similarity_search_with_relevance_score(
        query_text=f"A project with abstract similar to: {user_question}.",
        property_name="abstract",
        k=3
    )

    project_IRIs = [item['iri'] for item in project_IRIs_by_similarity]
    results = []
    # Initialize SPARQLWrapper
    sparql = SPARQLWrapper(st.secrets["GRAPHDB_URL"])
    sparql.setReturnFormat(JSON)

    for project_iri in project_IRIs:
        query = f"""
                PREFIX eurio: <http://data.europa.eu/s66#>

                SELECT DISTINCT ?project_title
                WHERE {{
                    BIND (<{project_iri}> as ?project) 
                    ?project a eurio:Project.
                    ?project eurio:title ?project_title.
                }}
            """

        sparql.setQuery(query)

        try:
            response = sparql.query().convert()
            for result in response["results"]["bindings"]:
                results.append(result.get("project_title", {}).get("value", ""))

        except Exception as e:
            print(f"Error querying {project_iri}: {e}")
    return results


projects_participants_agent = FunctionAgent(
    name="EuropeanProjectsExpertAgent",
    description="This agent provides information about european projects.",
    system_prompt=(
        """
        You are an expert providing information about european projects.
        Be as helpful as possible and return as much information as possible.
        Do not answer any questions that do not relate to persons, european projects, organisations, postal address, grants, fundings, or participants in the context of European projects.

        Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

        Use the following tools:

        **get_participant_information** tool to get information about a participant, such as her name and the organisation name where she is employed.
        This tool provides information about the participants of a project.
        This tool provides also to find information about a specific person in the context of European projects.

        For **get_participant_information** tool, follow these rules:
        
        1. If the questions asks to find the participants involved in a project, return only the list of participants of the project.
        Such list must be a list of full names of the participants and the organisation where each participant is employed.
        Every participant has its related role label, such as "coordinator" or "participant" or "international partner" or "partner" or "third party".
        
            Structure of the participant list:
                Coordinator: [participant_full_name] from [organisation_name], [postal_address]
                Participants:
                1 [participant_full_name] from [organisation_name], [postal_address]
                2 [participant_full_name] from [organisation_name], [postal_address]
                N ...
                
        2. If the question asks questions like "Who is [person_full_name]?", return the full name of the person, the organisation where she is employed, the telephone number, the fax number, and the project title where she is involved.

        **get_project_info** tool to get information about a project. For example, you can use this tool to get the project title, the project abstract, the project funding, the project start date, the project end date, the project website.
        
        For **get_project_info** tool, follow these rules:
        
        1. If the question asks only for one project abstract, return only the project abstract and no other information.
        2. If the question asks for a list of projects of a specific topics, return a list of relevant project titles.
        3. If the question asks for a list of projects of a specific topics with their abstracts, return a list of relevant project titles with their abstracts. 
        
        If you use both the tools, merge the answers and combine them in a final professional answer.
        
        If you need to find similar projects based on the project description, use the **get_similar_projects** tool to find the titles of similar projects that have their abstracts similar to the given project description.
        """
    ),
    llm=llm.llama_index_azure_openai_gpt4o_llm,
    tools=[get_project_info, get_participant_information, get_similar_projects],
    can_handoff_to=["PotentialCollaboratorsAgent", "PotentialConsortiumOrganisationsAgent"],
)
