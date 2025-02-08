from SPARQLWrapper import SPARQLWrapper, JSON
from langchain_community.vectorstores import Neo4jVector

from llama_index.core.agent.workflow import FunctionAgent

from rag.embeddings.neo4j_embedding_service import get_embedder

import streamlit as st

from rag.llm import get_llama_index_llm
from rag.tools.search_web import search_web

embedder = get_embedder()


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


async def get_similar_projects(project_description: str):
    "Finds similar project to a given project description"
    db = Neo4jVector.from_existing_graph(
        embedder,
        url=st.secrets["NEO4J_URI"],
        username=st.secrets["NEO4J_USERNAME"],
        password=st.secrets["NEO4J_PASSWORD"],
        index_name="abstractIndex",
        node_label="ns3__Project",
        text_node_properties=["ns3__abstract"],
        embedding_node_property="abstractEmbedding",
    )

    return db.similarity_search_with_relevance_scores(project_description, k=3)


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
        
        Once you have found the similar projects, run the following Cypher query to find the collaborators for each of the similar projects:
    
        MATCH (project:ns3__Project {title: 'PROJECT_TITLE'})
        MATCH (project)-[:ns3__hasInvolvedParty]->(org_role:ns3__OrganisationRole)
        MATCH (project)-[:ns3__hasInvolvedParty]->(per_role:ns3__PersonRole)
        MATCH (org_role)-[:ns3__isRoleOf]->(org:ns3__Organisation)
        MATCH (per_role)-[:ns3__isRoleOf]->(per:ns3__Person)
        MATCH (per_role)-[:ns3__isInvolvedIn]->(project)
        MATCH (per_role)-[:ns3__isEmployedBy]->(org)
        
        RETURN DISTINCT 
            per.rdfs__label AS person_full_name,
            org.rdfs__label AS organisation,
            project.ns3__title AS project_title;
        
        Finally, provide a list of potential collaborators for the given project description.
        Explain why you think these collaborators are relevant, highlighting that they have worked on similar found projects.
        Moreover, use the search_web tool to find more information about the potential collaborators and their research areas on the Web. 
        Make a list of keywords to highlight the research topics and areas for each suggested collaborator.

        Remember to maintain a professional and informative tone throughout your response. Your suggestions should be practical and directly applicable to someone looking for research collaboration.
        Avoid to thank for the given input, mention your knowledge source or provide any unnecessary information.
        """
    ),
    llm=get_llama_index_llm(),
    tools=[get_similar_projects, search_web],
    can_handoff_to=[],
)

# TODO: fare un ttool che prende i titoli dei progetti e poi runna la query per trovare i collaboratori