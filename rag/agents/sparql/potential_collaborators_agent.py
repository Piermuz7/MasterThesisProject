from SPARQLWrapper import SPARQLWrapper, JSON

from llama_index.core.agent.workflow import FunctionAgent
from pymongo import MongoClient

from rag.embeddings.mongo_db_embedding_store import EmbeddingStore

import streamlit as st

from rag.llm import get_llama_index_llm
from rag.tools.search_web import search_web

embedding_store = EmbeddingStore()

mongo_client = MongoClient(st.secrets["MONGO_URI"])
db = mongo_client[st.secrets["DB_NAME"]]
collection = db[st.secrets["COLLECTION_NAME"]]


async def get_collaborators_of_similar_projects(project_IRIs) -> any:
    "Get collaborators for a given project IRI."
    results = []

    # Initialize SPARQLWrapper
    sparql = SPARQLWrapper(st.secrets["GRAPHDB_URL"])
    sparql.setReturnFormat(JSON)

    for project_iri in project_IRIs:
        query = f"""
                PREFIX eurio: <http://data.europa.eu/s66#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

                SELECT DISTINCT ?person_full_name ?organisation ?project_title
                WHERE {{
                    BIND (<{project_iri}> as ?project) 
                    ?project a eurio:Project.
                    ?project eurio:title ?project_title.
                    ?project eurio:hasInvolvedParty ?party .
                    ?party a eurio:OrganisationRole.
                    ?party rdfs:label ?party_title.
                    ?party eurio:isRoleOf ?role .
                    ?role rdfs:label ?organisation.
                    ?person eurio:isInvolvedIn ?project.
                    ?person eurio:isEmployedBy ?role .
                    ?person eurio:isRoleOf ?p.
                    ?p rdfs:label ?person_full_name.
                }}
            """

        sparql.setQuery(query)

        try:
            response = sparql.query().convert()
            for result in response["results"]["bindings"]:
                results.append({
                    "full_name_person": result.get("person_full_name", {}).get("value", ""),
                    "organisation": result.get("organisation", {}).get("value", ""),
                    "project_title": result.get("project_title", {}).get("value", "")
                })

        except Exception as e:
            print(f"Error querying {project_iri}: {e}")

    return results


async def get_similar_projects(user_question: str) -> list[str]:
    "Finds similar project to a given project description"
    project_IRIs_by_similarity = embedding_store.find_similar_entities(
        query_text=f"A project with abstract similar to: {user_question}.",
        property_name="abstract",
        k=3
    )

    project_IRIs = [iri for iri, _ in project_IRIs_by_similarity]

    return project_IRIs


potential_collaborators_agent = FunctionAgent(
    name="PotentialCollaboratorsAgent",
    description="This agent suggest potential collaborators for a given project description.",
    system_prompt=(
        """
        You are an AI assistant tasked with suggesting potential collaborators for a given project description.
        Your goal is to provide a summary of the main concepts of the project description, and then a list of potential collaborators based on the project's content.
        
        Your response should include the name of the person, the organization where the person is employed, and the title of a project where the person was involved.
        
        You need to find similar projects based on the project description. Then, you need to find the collaborators for each of the similar projects.
        
        First of all use the get_similar_projects tool to find similar projects based on the project description. This tool will return a list of project IRIs.
        
        Once you have the project IRIs, use the get_collaborators_of_similar_projects tool to find the collaborators for each of the similar projects.
        
        Finally, provide a list of potential collaborators for the given project description.
        Explain why you think these collaborators are relevant, highlighting that they have worked on similar found projects.
        Moreover, use the search_web tool to find more information about the potential collaborators and their research areas on the Web. 
        Make a list of keywords to highlight the research topics and areas for each suggested collaborator.

        Remember to maintain a professional and informative tone throughout your response. Your suggestions should be practical and directly applicable to someone looking for research collaboration.
        Avoid to thank for the given input, mention your knowledge source or provide any unnecessary information.
        """
    ),
    llm=get_llama_index_llm(),
    tools=[get_similar_projects, get_collaborators_of_similar_projects, search_web],
    can_handoff_to=[],
)
