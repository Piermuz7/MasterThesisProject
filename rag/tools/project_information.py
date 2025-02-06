from langchain_community.chains.graph_qa.ontotext_graphdb import OntotextGraphDBQAChain
from langchain_core.prompts import PromptTemplate

from rag import llm
from rag.graph import graph


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

GRAPHDB_SPARQL_FIX_TEMPLATE = """
This following SPARQL query delimited by triple backticks
```
{generated_sparql}
```
is not valid.
The error delimited by triple backticks is
```
{error_message}
```
Give me a correct version of the SPARQL query.
Do not change the logic of the query.
Do not include any explanations or apologies in your responses.
Do not wrap the query in backticks.
Do not include any text except the SPARQL query generated.
The ontology schema delimited by triple backticks in Turtle format is:
```
{schema}
```
"""

GRAPHDB_SPARQL_FIX_PROMPT = PromptTemplate(
    input_variables=["error_message", "generated_sparql", "schema"],
    template=GRAPHDB_SPARQL_FIX_TEMPLATE,
)

GRAPHDB_QA_TEMPLATE = """Task: Generate a natural language response from the results of a SPARQL query.
You are an assistant that creates well-written and human understandable answers.
The information part contains the information provided, which you can use to construct an answer.
The information provided is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make your response sound like the information is coming from an AI assistant, but don't add any information.
Don't use internal knowledge to answer the question, just say you don't know if no information is available.
Information:
{context}

Question: {prompt}
Helpful Answer:"""
GRAPHDB_QA_PROMPT = PromptTemplate(
    input_variables=["context", "prompt"], template=GRAPHDB_QA_TEMPLATE
)

sparql_qa = OntotextGraphDBQAChain.from_llm(
    llm.get_llm(),
    graph=graph,
    sparql_generation_prompt=GRAPHDB_SPARQL_GENERATION_PROMPT,
    # sparql_fix_prompt=GRAPHDB_SPARQL_FIX_PROMPT,
    # qa_prompt=GRAPHDB_QA_PROMPT,
    verbose=True,
    allow_dangerous_requests=True
)

async def get_project_info(user_question: str) -> str:
    "Find the most relevant information about a given project"
    try:
        tool_input = {"query": user_question}
        return sparql_qa.invoke(input=tool_input)
    except Exception as e:
        return str(e)