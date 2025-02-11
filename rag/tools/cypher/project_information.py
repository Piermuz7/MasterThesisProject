from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts.prompt import PromptTemplate

from rag import llm
from rag.graph import neo4j_graph

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

1. Find title, URL, abstract, and status of a project named Topology driven methods for complex systems:"

    MATCH (project:ns3__Project)
    WHERE project.ns3__title = 'Topology driven methods for complex systems'
    RETURN DISTINCT 
        project.ns3__title AS title, 
        project.ns3__url AS url, 
        project.ns3__abstract AS abstract, 
        project.ns3__projectStatus AS status

2. Find title, URL, abstract, and status of a project named European Research Infrastructure on Highly Pathogenic Agents:"
    MATCH (project:ns3__Project)
    WHERE project.ns3__title = 'European Research Infrastructure on Highly Pathogenic Agents'
    RETURN DISTINCT 
        project.ns3__title AS title, 
        project.ns3__url AS url, 
        project.ns3__abstract AS abstract, 
        project.ns3__projectStatus AS status

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

chain = GraphCypherQAChain.from_llm(
    llm.langchain_anthropic_llm,
    graph=neo4j_graph,
    verbose=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    allow_dangerous_requests=True,
)


async def get_project_info(user_question: str) -> str:
    "Find the most relevant information about a given project"
    try:
        tool_input = {"query": user_question}
        return chain.invoke(input=tool_input)
    except Exception as e:
        return str(e)
