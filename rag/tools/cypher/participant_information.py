from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

from rag import llm
from rag.graph import neo4j_graph

examples = [
    {
        "question": "List the person name and organisation of the person who is involved in the project named 'Topology driven methods for complex systems':",
        "query": """
                    MATCH (project:ns3__Project {{ns3__title: 'Topology driven methods for complex systems'}})
                    MATCH (project)-[:ns3__hasInvolvedParty]->(org_role:ns3__OrganisationRole)
                    MATCH (project)-[:ns3__hasInvolvedParty]->(per_role:ns3__PersonRole)
                    MATCH (org_role)-[:ns3__isRoleOf]->(org:ns3__Organisation)
                    MATCH (per_role)-[:ns3__isRoleOf]->(per:ns3__Person)
                    MATCH (per_role)-[:ns3__isInvolvedIn]->(project)
                    MATCH (per_role)-[:ns3__isEmployedBy]->(org)
                    
                    RETURN DISTINCT 
                        per.rdfs__label AS person_full_name,
                        org.rdfs__label AS organisation
                """
    },
    {
        "question": "Find the number of participants who is involved in the project named 'Topology driven methods for complex systems':",
        "query": """
                    MATCH (project:ns3__Project {{ns3__title: 'Topology driven methods for complex systems'}})
                    MATCH (project)-[:ns3__hasInvolvedParty]->(org_role:ns3__OrganisationRole)
                    MATCH (project)-[:ns3__hasInvolvedParty]->(per_role:ns3__PersonRole)
                    MATCH (org_role)-[:ns3__isRoleOf]->(org:ns3__Organisation)
                    MATCH (per_role)-[:ns3__isRoleOf]->(per:ns3__Person)
                    MATCH (per_role)-[:ns3__isInvolvedIn]->(project)
                    MATCH (per_role)-[:ns3__isEmployedBy]->(org)
                    
                    RETURN COUNT(DISTINCT per) AS participant_count
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

The ontology schema is:
{schema}

Use only the classes and properties provided in the schema to construct the Cypher query.
Do not use any classes or properties that are not explicitly provided in the Cypher query.
Include all necessary prefixes.
Do not include any explanations or apologies in your responses.
Do not wrap the query in backticks.
Do not include any text except the Cypher query generated.

Follow these rules:
    1.  If the question asks to "find" or "list" participants, use Example 1.
        Provide a list of participants with their names and organisations.
        Format each entry as:
        - [person_full_name] ([organisation])

    2.  If the question asks to "count" or "determine the number of" participants, use Example 2.

Given an input question, create a syntactically very accurate Cypher query based on the following examples:
"""

prompt_part2 = """
The question is:
{prompt}
"""

CYPHER_GENERATION_TEMPLATE = prompt_part1 + formatted_examples + prompt_part2

Cypher_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "prompt"],
    template=CYPHER_GENERATION_TEMPLATE,
)

cypher_qa = GraphCypherQAChain.from_llm(
    llm=llm.get_langchain_llm(),
    graph=neo4j_graph,
    cypher_prompt=Cypher_GENERATION_PROMPT,
    allow_dangerous_requests=True,
    verbose=True
)


async def get_participant_information(user_question: str) -> str:
    "List the person name and organisation of the person who is involved in a given project title"
    try:
        tool_input = {"query": user_question}
        return cypher_qa.invoke(input=tool_input)
    except Exception as e:
        return str(e)
