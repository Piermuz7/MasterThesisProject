from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts.prompt import PromptTemplate

from rag import llm
from rag.graph import neo4j_graph

import streamlit as st

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Examples: Here are a few examples of generated Cypher statements for particular questions:

1. List the person name and organisation of the person who is involved in the project named 'Topology driven methods for complex systems':
    
    MATCH (project:ns3__Project {{ns3__title: 'Topology driven methods for complex systems'}})
    MATCH (project)-[:ns3__hasInvolvedParty]->(org_role:ns3__OrganisationRole)
    MATCH (project)-[:ns3__hasInvolvedParty]->(per_role:ns3__PersonRole)
    MATCH (org_role)-[:ns3__isRoleOf]->(org:ns3__Organisation)
    MATCH (per_role)-[:ns3__isRoleOf]->(per:ns3__Person)
    MATCH (per_role)-[:ns3__isInvolvedIn]->(project)
    MATCH (per_role)-[:ns3__isEmployedBy]->(org)
    
    RETURN DISTINCT 
        per.rdfs__label AS person_full_name,
        org.rdfs__label AS organisation_name
        
        
2. Find the number of participants who is involved in the project named 'Topology driven methods for complex systems':

    MATCH (project:ns3__Project {{ns3__title: 'Topology driven methods for complex systems'}})
    MATCH (project)-[:ns3__hasInvolvedParty]->(org_role:ns3__OrganisationRole)
    MATCH (project)-[:ns3__hasInvolvedParty]->(per_role:ns3__PersonRole)
    MATCH (org_role)-[:ns3__isRoleOf]->(org:ns3__Organisation)
    MATCH (per_role)-[:ns3__isRoleOf]->(per:ns3__Person)
    MATCH (per_role)-[:ns3__isInvolvedIn]->(project)
    MATCH (per_role)-[:ns3__isEmployedBy]->(org)
    
    RETURN COUNT(DISTINCT per) AS participant_count


The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "prompt"],
    template=CYPHER_GENERATION_TEMPLATE,
)

chain = GraphCypherQAChain.from_llm(
    llm.get_langchain_anthropic_llm(),
    graph=neo4j_graph,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    allow_dangerous_requests=True,
    verbose=True
)


async def get_participant_information(user_question: str) -> str:
    "List the person name and organisation name of the person who is involved in a given project title"
    try:
        tool_input = {"query": user_question}
        return chain.invoke(input=tool_input)
    except Exception as e:
        return str(e)
