from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector

from llama_index.core.agent.workflow import FunctionAgent

from rag.embeddings.neo4j_embedding_service import get_embedder

import streamlit as st

from rag.tools.search_web import search_web

from rag import llm

embedder = get_embedder()


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

    x = db.similarity_search_with_relevance_scores(project_description, k=3)
    return x


async def get_collaborators_of_similar_projects(project_URIs) -> any:
    "Get collaborators for a given project URI."
    results = []

    driver = GraphDatabase.driver(st.secrets["NEO4J_URI"],
                                  auth=(st.secrets["NEO4J_USERNAME"], st.secrets["NEO4J_PASSWORD"]))

    for project_uri in project_URIs:

        with driver.session() as session:
            query = f"""
                    MATCH (project)-[:ns3__hasInvolvedParty]->(org_role:ns3__OrganisationRole)
                    MATCH (project)-[:ns3__hasInvolvedParty]->(per_role:ns3__PersonRole)
                    MATCH (org_role)-[:ns3__isRoleOf]->(org:ns3__Organisation)
                    MATCH (per_role)-[:ns3__isRoleOf]->(per:ns3__Person)
                    MATCH (per_role)-[:ns3__isInvolvedIn]->(project)
                    MATCH (per_role)-[:ns3__isEmployedBy]->(org)
                    WHERE project.uri = '{project_uri}'
                    RETURN DISTINCT 
                        per.rdfs__label AS person_full_name,
                        org.rdfs__label AS organisation,
                        project.ns3__title AS project_title;
                """

            response = session.run(query).data()
            for record in response:
                results.append({
                    "full_name_person": record.get("person_full_name", ""),
                    "organisation": record.get("organisation", ""),
                    "project_title": record.get("project_title", "")
                })

    return results


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
        
        Once you have found the similar projects, use the get_collaborators_of_similar_projects tool to find the collaborators for each of the similar projects.
        This tool is used to find the collaborators for a given project URI.
        
        Finally, provide a list of potential collaborators for the given project description.
        Explain why you think these collaborators are relevant, highlighting that they have worked on similar found projects.
        
        Only when you have got the list of potential collaborators, use the search_web tool to find more information about the potential collaborators and their research areas on the Web. 
        Make a list of keywords to highlight the research topics and areas for each suggested collaborator.

        If you did not find any similar projects or collaborators, do not use the search_web tool.
        In this case, provide a professional response explaining that you did not find any similar projects or collaborators.

        Remember to maintain a professional and informative tone throughout your response. Your suggestions should be practical and directly applicable to someone looking for research collaboration.
        Avoid to thank for the given input, mention your knowledge source or provide any unnecessary information.
        """
    ),
    llm=llm.get_llama_index_anthropic_llm(),
    tools=[get_similar_projects, get_collaborators_of_similar_projects, search_web],
    can_handoff_to=[],
)

# TODO: fare un ttool che prende i titoli dei progetti e poi runna la query per trovare i collaboratori
'''
llm = Gemini(
    model="models/gemini-2.0-flash",
    api_key=st.secrets["api_key"]["GOOGLE_KEY"],
),
'''
