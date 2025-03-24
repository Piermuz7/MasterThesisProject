from langchain_community.chains.graph_qa.ontotext_graphdb import OntotextGraphDBQAChain
from langchain_core.prompts import PromptTemplate

from graphrag import llm
from graphrag.graph import graph_db

examples = [
    {
        "question": "List the person name and organisation of the person who is involved in the project named 'Topology driven methods for complex systems':",
        "query": """
                    PREFIX eurio: <http://data.europa.eu/s66#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    
                    SELECT DISTINCT ?person_full_name ?organisation ?postal_address ?role_label
                    WHERE {{
                        ?project a eurio:Project .
                        ?project eurio:title "Topology driven methods for complex systems".
                        ?project eurio:hasInvolvedParty ?party .
                        ?party a eurio:OrganisationRole.
                        ?party rdfs:label ?party_title.
                        ?party eurio:roleLabel ?role_label.
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
                    ORDER BY ?role_label
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
    },
    {
        "question": "Which project was Emanuela Merelli involved in?",
        "query": """
                    ```
                    PREFIX eurio: <http://data.europa.eu/s66#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    
                    SELECT DISTINCT ?project_title
                    WHERE {{
                        ?person a eurio:Person .
                        ?person rdfs:label ?label .
                        FILTER (LCASE(?label) = "emanuela merelli")
                        ?person eurio:hasRole ?role.
                        ?role eurio:isInvolvedIn ?project .
                        ?project eurio:title ?project_title .
                    }}
                    ```
                """
    },
    {
        "question": "What is the postal address of university of Camerino?",
        "query": """
                    ```
                    PREFIX eurio: <http://data.europa.eu/s66#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    
                    SELECT DISTINCT ?postal_address
                    WHERE {{
                        ?organisation a eurio:Organisation .
                        ?organisation rdfs:label ?label .
                        FILTER(REGEX(?label, "universita degli studi di camerino", "i")) .
                        ?organisation eurio:hasSite ?site .
                        ?site eurio:hasAddress ?address .
                        ?address eurio:fullAddress ?postal_address .
                    }}
                    ```
                """
    },
    {
        "question": "What is the postal address of university of Camerino?",
        "query": """
                    ```
                    PREFIX eurio: <http://data.europa.eu/s66#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    
                    SELECT DISTINCT ?postal_address
                    WHERE {{
                        ?organisation a eurio:Organisation .
                        ?organisation rdfs:label ?label .
                        FILTER(CONTAINS(LCASE(?label), "camerino")) .
                        ?organisation eurio:hasSite ?site .
                        ?site eurio:hasAddress ?address .
                        ?address eurio:fullAddress ?postal_address .
                    }}
                    ```
                """
    },
    {
        "question": "Who is Emanuela Merelli?",
        "query": """
                    ```
                    PREFIX eurio: <http://data.europa.eu/s66#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    
                    SELECT ?person_label ?org_name ?telephone ?fax ?project_title
                    WHERE {{
                        ?person a eurio:Person .
                        ?person rdfs:label ?person_label .
                        ?person eurio:hasContactDetails ?contact_details.
                        ?contact_details eurio:telephone ?telephone.
                        OPTIONAL{{?contact_details eurio:faxNumber ?fax.}}
                        FILTER (LCASE(?person_label) = "emanuela merelli") .
                        ?person eurio:hasRole ?role.
                        ?role eurio:isEmployedBy ?org.
                        ?org rdfs:label ?org_name.
                        ?role eurio:isInvolvedIn ?project.
                        ?project a eurio:Project.
                        ?project eurio:title ?project_title.
                    }}
                    ```
                """
    },
    {
        "question": "Provide information about Emanuela Merelli",
        "query": """
                    ```
                    PREFIX eurio: <http://data.europa.eu/s66#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

                    SELECT ?person_label ?org_name ?telephone ?fax ?project_title
                    WHERE {{
                        ?person a eurio:Person .
                        ?person rdfs:label ?person_label .
                        ?person eurio:hasContactDetails ?contact_details.
                        ?contact_details eurio:telephone ?telephone.
                        OPTIONAL{{?contact_details eurio:faxNumber ?fax.}}
                        FILTER (LCASE(?person_label) = "emanuela merelli") .
                        ?person eurio:hasRole ?role.
                        ?role eurio:isEmployedBy ?org.
                        ?org rdfs:label ?org_name.
                        ?role eurio:isInvolvedIn ?project.
                        ?project a eurio:Project.
                        ?project eurio:title ?project_title.
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

Follow these rules:
    1.  If the question asks to "find" or "list" participants, use Example 1.
        Provide a list of participants with their names and organisations.
        Format each entry as:
        - [person_full_name] ([organisation]), [postal_address] , ([role_label])

    2.  If the question asks to "count" or "determine the number of" participants, use Example 2.
    
    3.  If the question asks "which project was [person_full_name] involved in", use Example 3.
    
    4.  If the question asks "what is the postal address of [organisation_name]", use Example 4 or 5.
    
    5.  If the question asks "who is [person name]" or "provide information about [person name]", use Example 6 or 7.
        In this case, you must use the eurio:telephone and eurio:faxNumber properties, not the eurio:telephoneNumber and eurio:fax properties.

Given an input question, create a syntactically very accurate SPARQL query based on the following examples:
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


