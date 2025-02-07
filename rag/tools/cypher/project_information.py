from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

from rag import llm
from rag.graph import neo4j_graph

examples = [
    {
        "question": "Find title, URL, abstract, and status of a project named Topology driven methods for complex systems:",
        "query": """
                    ```
                    MATCH (project:ns3__Project)
                    WHERE project.ns3__title = 'Topology driven methods for complex systems'
                    RETURN DISTINCT 
                        project.ns3__title AS title, 
                        project.ns3__url AS url, 
                        project.ns3__abstract AS abstract, 
                        project.ns3__projectStatus AS status
                    ```
                """
    },
    {
        "question": "Find title, URL, abstract, and status of a project named European Research Infrastructure on Highly Pathogenic Agents:",
        "query": """
                    ```
                    MATCH (project:ns3__Project)
                    WHERE project.ns3__title = 'European Research Infrastructure on Highly Pathogenic Agents'
                    RETURN DISTINCT 
                        project.ns3__title AS title, 
                        project.ns3__url AS url, 
                        project.ns3__abstract AS abstract, 
                        project.ns3__projectStatus AS status
                    ```
                """
    }
]

formatted_examples = "\n\n".join([
    f"Example {i + 1}:\nQuestion: {ex['question']}\nCypher Query: {ex['query']}"
    for i, ex in enumerate(examples)
])

prompt_part1 = """
You are a Cypher expert about the EURIO graph database.
Your task is to generate Cypher statements to query the graph database.

The ontology schema delimited by triple backticks is:
{schema}

Use only the classes and properties provided in the schema to construct the Cypher query.
Do not use any classes or properties that are not explicitly provided in the Cypher query.
Include all necessary prefixes.
Do not include any explanations or apologies in your responses.
Do not wrap the query in backticks.
Do not include any text except the Cypher query generated.

Given an input question, create a syntactically very accurate Cypher query based on the following examples :
"""
prompt_part2 = """
The question is:
{prompt}
"""

CYPHER_GENERATION_TEMPLATE = prompt_part1 + formatted_examples + prompt_part2

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "prompt"],
    template=CYPHER_GENERATION_TEMPLATE,
)

cypher_qa = GraphCypherQAChain.from_llm(
    llm=llm.get_langchain_llm(),
    graph=neo4j_graph,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    verbose=True,
    allow_dangerous_requests=True
)


async def get_project_info(user_question: str) -> str:
    "Find the most relevant information about a given project"
    try:
        tool_input = {"query": user_question}
        return cypher_qa.invoke(input=tool_input)
    except Exception as e:
        return str(e)
