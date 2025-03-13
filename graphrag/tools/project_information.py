from langchain_community.chains.graph_qa.ontotext_graphdb import OntotextGraphDBQAChain
from langchain_core.prompts import PromptTemplate

from graphrag import llm
from graphrag.graph import graph_db

examples = [
    {
        "question": "Find title, URL, abstract, and status of a project named Knowledge-Based Information Agent with Social Competence and Human Interaction Capabilities:",
        "query": """
                    ```
                    PREFIX eurio: <http://data.europa.eu/s66#>
                    SELECT DISTINCT ?title ?url ?abstract ?status
                    WHERE{{
                        ?project a eurio:Project .
                        ?project eurio:title "Knowledge-Based Information Agent with Social Competence and Human Interaction Capabilities" .
                        ?project eurio:title ?title.
                        ?project eurio:url ?url.
                        ?project eurio:abstract ?abstract.
                        ?project eurio:projectStatus ?status.
                    }}
                    ```
                """
    },
    {
        "question": "Find title, URL, abstract, and status of a project named BIM-based holistic tools for Energy-driven Renovation of existing Residences:",
        "query": """
                    ```
                    PREFIX eurio: <http://data.europa.eu/s66#>
                    SELECT DISTINCT ?title ?url ?abstract ?status
                    WHERE{{
                        ?project a eurio:Project .
                        ?project eurio:title "BIM-based holistic tools for Energy-driven Renovation of existing Residences" .
                        ?project eurio:title ?title.
                        ?project eurio:url ?url.
                        ?project eurio:abstract ?abstract.
                        ?project eurio:projectStatus ?status.
                    }}
                    ```
                """
    }
]

formatted_examples = "\n\n".join([
    f"Example {i + 1}:\nQuestion: {ex['question']}\nSPARQL Query: {ex['query']}"
    for i, ex in enumerate(examples)
])

prompt_part1 = """
You are a SPARQL expert about the EURIO graph database.
The ontology schema delimited by triple backticks in Turtle format is:
```
{schema}
```

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a SPARQL statement.
Do not include any text except the generated SPARQL statement.

Use only the classes and properties provided in the schema to construct the SPARQL query.
Do not use any classes or properties that are not explicitly provided in the SPARQL query.
Include all necessary prefixes.
Do not include any explanations or apologies in your responses.
Do not wrap the query in backticks.
Do not include any text except the SPARQL query generated.

Given an input question, create a syntactically very accurate SPARQL query based on the following examples delimited by triple backticks:
"""
prompt_part2 = """
The question delimited by triple backticks is:
```
{prompt}
```
"""

GRAPHDB_SPARQL_GENERATION_TEMPLATE = prompt_part1 + formatted_examples + prompt_part2

GRAPHDB_SPARQL_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "prompt"],
    template=GRAPHDB_SPARQL_GENERATION_TEMPLATE,
)

sparql_qa = OntotextGraphDBQAChain.from_llm(
    llm=llm.langchain_azure_openai_gpt4o_llm,
    graph=graph_db,
    sparql_generation_prompt=GRAPHDB_SPARQL_GENERATION_PROMPT,
    verbose=True,
    allow_dangerous_requests=True
)


async def get_project_info(user_question: str) -> str:
    "Find the most relevant information about a given project"
    try:
        tool_input = {"query": user_question}
        print(f"Querying {user_question}...")
        return sparql_qa.invoke(input=tool_input)
    except Exception as e:
        return str(e)
