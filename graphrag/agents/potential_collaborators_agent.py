from SPARQLWrapper import SPARQLWrapper, JSON

from llama_index.core.agent.workflow import FunctionAgent

from graphrag.embeddings.graph_embedding_service import GraphEmbeddingStore

import streamlit as st

from graphrag import llm
from graphrag.tools.search_web import search_web

embedding_store = GraphEmbeddingStore()


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

                SELECT DISTINCT ?person_full_name ?organisation ?postal_address ?project_title
                WHERE {{
                    BIND (<{project_iri}> as ?project) 
                    ?project a eurio:Project.
                    ?project eurio:title ?project_title.
                    ?project eurio:hasInvolvedParty ?party .
                    ?party a eurio:OrganisationRole.
                    ?party rdfs:label ?party_title.
                    ?party eurio:isRoleOf ?role .
                    ?role rdfs:label ?organisation.
                    ?role eurio:hasSite ?site.
                    ?site eurio:hasAddress ?address.
                    ?address eurio:fullAddress ?postal_address.
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
                    "postal_address": result.get("postal_address", {}).get("value", ""),
                    "project_title": result.get("project_title", {}).get("value", "")
                })

        except Exception as e:
            print(f"Error querying {project_iri}: {e}")

    return results


async def get_similar_projects(user_question: str) -> list[str]:
    "Finds similar project to a given project description"
    project_IRIs_by_similarity = embedding_store.similarity_search_with_relevance_score(
        query_text=f"A project with abstract similar to: {user_question}.",
        property_name="abstract",
        k=3
    )

    project_IRIs = [item['iri'] for item in project_IRIs_by_similarity]

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
        
        First of all use the get_similar_projects tool to find similar projects that have their abstracts similar to the given project description.
        This tool is used to find similar projects and to consider all the relevant information about a project such as the title, abstract, the uri, and other details.
        The uri of each project is used to find the collaborators for each of the similar projects.
        
        Once you have the project IRIs, use the get_collaborators_of_similar_projects tool to find the collaborators for each of the found similar projects.
        This tool is used to find the collaborators for a given project URI.
        You must consider and use all the project IRIs returned by the get_similar_projects tool.
        
        Finally, provide a list of potential collaborators for the given project description.
        Explain why you think these collaborators are relevant, highlighting that they have worked on similar found projects.
        
        Only when you have got the list of potential collaborators, use the search_web tool to find more information about the potential collaborators and their research areas on the Web. 
        Make a list of keywords to highlight the research topics and areas for each suggested collaborator.
        
        Structure of the potential collaborators list:
        
        1.  [person_full_name]
            
            • Organisation: [organisation_name], [postal_address]
            
            • Project title: [project_title]
            
            • Research areas: [research_areas]
            
        2.  [person_full_name]
            
            • Organisation: [organisation_name], [postal_address]
            
            • Project title: [project_title]
            
            • Research areas: [research_areas]
            
        N.  ...

        If you did not find any similar projects or collaborators, do not use the search_web tool.
        In this case, provide a professional response explaining that you did not find any similar projects or collaborators.

        Remember to maintain a professional and informative tone throughout your response. Your suggestions should be practical and directly applicable to someone looking for research collaboration.
        Avoid to thank for the given input, mention your knowledge source or provide any unnecessary information.
        """
    ),
    llm=llm.llama_index_azure_openai_llm,
    tools=[get_similar_projects, get_collaborators_of_similar_projects, search_web],
    can_handoff_to=[],
)