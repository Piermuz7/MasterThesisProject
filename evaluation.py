from datasets import Dataset
from langchain_community.chains.graph_qa.ontotext_graphdb import OntotextGraphDBQAChain

from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jVector, GraphCypherQAChain

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, \
    answer_similarity, answer_correctness
from ragas.metrics._aspect_critic import harmfulness

from graphrag.embeddings.neo4j_embedding_service import get_embedder
from graphrag.embeddings.mongo_db_embedding_store import EmbeddingStore
from graphrag.graph import neo4j_graph, graph_db
import graphrag.llm as llm

from graphrag.embeddings import mongo_db_embedding_store

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

embedder = get_embedder()

llm = llm.langchain_azure_openai_llm

vector_store = Neo4jVector.from_existing_graph(
    embedder,
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    index_name="titleIndex",
    node_label="ns3__Project",
    text_node_properties=["ns3__title"],
    embedding_node_property="titleEmbedding",
)

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

Do not modify the query of the user.
For example, if the user give you the project title like "Innovative non- intrusive laser gas sensors on food production for real time quality/safety  in line control of food packaging and bottling systems", do not remove spaces or change the order of the words.

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

cypher_qa = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=neo4j_graph,
    verbose=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    allow_dangerous_requests=True,
)

SPARQL_GENERATION_TEMPLATE = """
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

    1. "Find title, URL, abstract, and status of a project named Knowledge-Based Information Agent with Social Competence and Human Interaction Capabilities:",
    
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
    
    2. Find title, URL, abstract, and status of a project named BIM-based holistic tools for Energy-driven Renovation of existing Residences:",
        
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
        
    The question delimited by triple backticks is:
    ```
    {prompt}
    ```
"""

SPARQL_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "prompt"], template=SPARQL_GENERATION_TEMPLATE
)

sparql_qa = OntotextGraphDBQAChain.from_llm(
    llm=llm,
    graph=graph_db,
    verbose=True,
    sparql_generation_prompt=SPARQL_GENERATION_PROMPT,
    allow_dangerous_requests=True,
)

queries = [
    "What is the abstract of a project named 'IMproving Preparedness and RIsk maNagemenT for flash floods and debriS flow events'?",
    "What are the general information of the project named 'Topology driven methods for complex systems'?",
    "5 title of projects with abstract similar to: Many complex systems are characterized by multi-level properties that make the study of their dynamics and of their emerging phenomena a daunting task. The huge amount of data available in modern sciences can be expected to support great progress in these studies, even though the nature of the data varies. Given that, it is crucial to extract as much as possible features from data, including qualitative (topological) ones. The goal of this project is to provide methods driven by the topology of data for describing the dynamics of multi-level complex systems. To this end the project will develop new mathematical and computational formalisms accounting for topological effects. To pursue these objectives the project brings together scientists from many diverse fields including as topology and geometry, statistical physics and information theory, computer science and biology. The proposed methods, obtained through concerted efforts, will cover different aspects of the science of complexity ranging from foundations, to simulations through modelling and analysis, and are expected to constitute the building blocks for a new generalized theory of complexity.",
]

ground_truths = [
    """
    The aim of IMPRINTS is to contribute to reduce loss of life and economic damage through the improvement of the preparedness and the operational risk management for Flash Flood and Debris Flow [FF/DF] generating events, as well as to contribute to sustainable development through reducing damages to the environment. To achieve this ultimate objective the project is oriented to produce methods and tools to be used by emergency agencies and utility companies responsible for the management of FF/DF risks and associated effects. Impacts of future changes, including climatic, land use and socioeconomic will be analysed in order to provide guidelines for mitigation and adaptation measures. Specifically, the consortium will develop an integrated probabilistic forecasting FF/ DF system as well as a probabilistic early warning and a rule-based probabilistic forecasting system adapted to the operational use by practitioners. These systems will be tested on five selected flash flood prone areas, two located in mountainous catchments in the Alps, and three in Mediterranean catchments. The IMPRINTS practitioner partners, risk management authorities and utility company managers in duty of emergency management in these areas, will supervise these tests. The development of such systems will be carried out using and capitalising the results of previous and ongoing research on FF/DF forecasting and warning systems, in which several of the partners have played a prominent role. One major result of the project will be a operational prototype including the tools and methodologies developed under the project. This prototype will be designed under the premise of its ultimate commercialization and use worldwide. The consortium, covering all the actors involved in the complex chain of FF & DF forecasting, has been carefully selected to ensure the achievement of this. Specific actions to exploit and protect the results and the intellectual property of the partners have been also defined.
    """,
    """
    The project focused on studying complex systems with multi-level properties, specifically developing methods for analyzing these systems using data topology.
    
    Key objectives included:
	•	Extracting features from large amounts of data, focusing on topological characteristics
	•	Developing new mathematical and computational methods
	•	Understanding the dynamics of multi-level complex systems
	
	The project brought together experts from various fields, including:
	•	Topology and geometry
	•	Statistical physics
	•	Information theory
	•	Computer science
	•	Biology
	
	Participating Researchers and Institutions:

The project involved collaboration among several prestigious institutions and researchers:
	1.	CENTRE NATIONAL DE LA RECHERCHE SCIENTIFIQUE (CNRS)
	    •	Béatrice SAINT-CRICQ
	2.	UNIVERSITE D’AIX MARSEILLE
	    •	Céline Damon
	3.	UNIVERSITEIT VAN AMSTERDAM
	    •	Vanessa Wolters
	4.	SYDDANSK UNIVERSITET
	    •	Christian Reidys
	5.	UNIVERSITA DEGLI STUDI DI CAMERINO
	    •	Emanuela Merelli
	6.	ISTITUTO PER L’INTERSCAMBIO SCIENTIFICO
	    •	Roberto Palermo
	
    """,
    """
    Topology driven methods for complex systems, Topological Complex Systems, Understanding Random Systems via Algebraic Topology, Self-Organised information PrOcessing, CriticaLity and Emergence in multilevel Systems, Hierarchical Analysis of Complex Dynamical Systems.
    """
]

results = []
contexts = []
'''
for query in queries:
    graph_result = cypher_qa.invoke(input={"query": query})

    vector_result = vector_store.similarity_search(query, top_k=5)

    final_prompt = f"""
    You are an expert providing information about european projects.
    Be as helpful as possible and return as much useful information as possible.
    Do not adding any additional useless information to the answer, for example do not answer that you are happy to help, do not answer you are sorry for the delay.
    Do not answer any questions that do not relate to projects, organisations, grants, fundings, or participants.
    Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

    Answer only the question asked by the user. Do not provide any additional information that is not asked by the user.

    Follow these rules:

    1. If the user asks for the abstract of a project, provide only the abstract of the project.


    Your task is to analyze and synthesize information from two sources: the top result from a similarity search (unstructured information) and relevant data from a graph database (structured information).
    Given the user's query: {query}, provide a meaningful and efficient answer based on the insights derived from the following data:

    Unstructured information: {vector_result}.
    Structured information: {graph_result}
    """
    res = llm.invoke(input=final_prompt)
    print(res)
    result = res.content
    print(result)
    results.append(result)
    sources = vector_result
    print("rources: ", sources)
    contents = []
    for s in sources:
        contents.append(s.page_content)
    contexts.append(contents)
'''


embedding_store = EmbeddingStore()
for query in queries:
    graph_result = sparql_qa.invoke(input={"query": query})

    vector_result = embedding_store.similarity_search_with_relevance_score(
    query_text=query,
    property_name="title",
    k=3
)

    final_prompt = f"""
    You are an expert providing information about european projects.
    Be as helpful as possible and return as much useful information as possible.
    Do not adding any additional useless information to the answer, for example do not answer that you are happy to help, do not answer you are sorry for the delay.
    Do not answer any questions that do not relate to projects, organisations, grants, fundings, or participants.
    Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

    Answer only the question asked by the user. Do not provide any additional information that is not asked by the user.

    Follow these rules:

    1. If the user asks for the abstract of a project, provide only the abstract of the project.


    Your task is to analyze and synthesize information from two sources: the top result from a similarity search (unstructured information) and relevant data from a graph database (structured information).
    Given the user's query: {query}, provide a meaningful and efficient answer based on the insights derived from the following data:

    Unstructured information: {vector_result}.
    Structured information: {graph_result}
    """
    res = llm.invoke(input=final_prompt)
    print(res)
    result = res.content
    print(result)
    results.append(result)
    sources = vector_result
    print("rources: ", sources)
    contents = []
    for s in sources:
        print(s['entity']['label'])
        contents.append(s['entity']['label'])
    contexts.append(contents)

d = {
    "user_input": queries,
    "retrieved_contexts": contexts,
    "response": results,
    "reference": ground_truths,
}

dataset = Dataset.from_dict(d)
score = evaluate(dataset,
                 metrics=[faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall,
                          answer_similarity, answer_correctness, harmfulness],
                 llm=llm,
                 embeddings=embedder
                 )
score_df = score.to_pandas()
score_df.to_csv("evaluation/evaluation_scores_openai.csv", encoding="utf-8", index=False)
