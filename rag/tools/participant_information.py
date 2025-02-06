from langchain_community.chains.graph_qa.ontotext_graphdb import OntotextGraphDBQAChain
from langchain_core.prompts import PromptTemplate

from rag import llm
from rag.graph import graph

examples = [
    {
        "question": "List the person name and organisation of the person who is involved in the project named 'Topology driven methods for complex systems':",
        "query": """
                    PREFIX eurio: <http://data.europa.eu/s66#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    
                    SELECT DISTINCT ?person_full_name ?organisation
                    WHERE {{
                        ?project a eurio:Project .
                        ?project eurio:title "Topology driven methods for complex systems".
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
    },
    {
        "question": "Find the number of participants who is involved in the project named 'Topology driven methods for complex systems':",
        "query": """
                    ```
                    PREFIX eurio: <http://data.europa.eu/s66#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    
                    SELECT (COUNT(DISTINCT ?person_full_name) AS ?participant_count)
                    WHERE {{
                        ?project a eurio:Project .
                        ?project eurio:title "Topology driven methods for complex systems".
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

Follow these rules:
    1.  If the question asks to "find" or "list" participants, use Example 1.
        Provide a list of participants with their names and organisations.
        Format each entry as:
        - [person_full_name] ([organisation])

    2.  If the question asks to "count" or "determine the number of" participants, use Example 2.

Given an input question, create a syntactically very accurate SPARQL query based on the following examples:
"""

prompt_part2 = """
The question delimited by triple backticks is:
```
{prompt}
```
"""

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

GRAPHDB_SPARQL_GENERATION_TEMPLATE = prompt_part1 + formatted_examples + prompt_part2

GRAPHDB_SPARQL_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "prompt"],
    template=GRAPHDB_SPARQL_GENERATION_TEMPLATE,
)

sparql_qa = OntotextGraphDBQAChain.from_llm(
    llm.get_llm(),
    graph=graph,
    sparql_generation_prompt=GRAPHDB_SPARQL_GENERATION_PROMPT,
    qa_template=GRAPHDB_QA_TEMPLATE,
    allow_dangerous_requests=True,
    verbose=True
)


async def get_participant_information(user_question: str) -> str:
    "List the person name and organisation of the person who is involved in a given project title"
    try:
        tool_input = {"query": user_question}
        return sparql_qa.invoke(input=tool_input)
    except Exception as e:
        return str(e)
