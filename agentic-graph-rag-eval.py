import glob
import asyncio

import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
from datasets import Dataset
from langchain_ollama import OllamaEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, \
    answer_similarity, answer_correctness
from graphrag.agent_workflow import execute_agent_workflow
from graphrag.agents.projects_participants_agent import embedding_store
import graphrag.llm as grag_llm
from evaluation.qa import potential_consortium_queries, potential_consortium_ground_truths
import os
import streamlit as st

os.environ["TOKENIZERS_PARALLELISM"] = "false"

llm = grag_llm.langchain_azure_openai_gpt4o_llm
embedder = OllamaEmbeddings(model="all-minilm:l6-v2")

results = []
contexts = []
cleaned_context = []

# Evaluation queries. For the evaluation, we use the potential consortium queries.
all_queries = [{"potential_consortium": potential_consortium_queries}]  # ,{"titles": []}, {"abstracts": []}]


def get_organisations_of_similar_projects(project_IRIs) -> any:
    "Get organisations for a given project IRI."
    results = []

    # Initialize SPARQLWrapper
    sparql = SPARQLWrapper(st.secrets["GRAPHDB_URL"])
    sparql.setReturnFormat(JSON)

    for project_iri in project_IRIs:
        query = f"""
                PREFIX eurio: <http://data.europa.eu/s66#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

                SELECT DISTINCT ?organisation ?postal_address ?project_title ?project_abstract
                WHERE {{
                    BIND (<{project_iri}> as ?project) 
                    ?project a eurio:Project.
                    ?project eurio:title ?project_title.
                    ?project eurio:abstract ?project_abstract.
                    ?project eurio:hasInvolvedParty ?party .
                    ?party a eurio:OrganisationRole.
                    ?party rdfs:label ?party_title.
                    ?party eurio:isRoleOf ?role .
                    ?role rdfs:label ?organisation.
                    ?role eurio:hasSite ?site.
                    ?site eurio:hasAddress ?address.
                    ?address eurio:fullAddress ?postal_address.
                }}
                ORDER BY ?organisation
            """

        sparql.setQuery(query)

        try:
            response = sparql.query().convert()
            for result in response["results"]["bindings"]:
                results.append({
                    "organisation": result.get("organisation", {}).get("value", ""),
                    "postal_address": result.get("postal_address", {}).get("value", ""),
                    "project_title": result.get("project_title", {}).get("value", "")
                })

        except Exception as e:
            print(f"Error querying {project_iri}: {e}")

    return results


def format_value(value):
    """
    Returns the string representation of a value.
    If the value is a list, it joins the items with commas.
    """
    if isinstance(value, list):
        return ", ".join(format_value(item) for item in value)
    else:
        return str(value)


def project_info_to_str(d):
    """
    Converts a dictionary to a string in the format:
    organisation_value. postal_address_value. project_title_value
    If any of these keys are missing, an empty string is used.
    """
    org = format_value(d.get("organisation", ""))
    addr = format_value(d.get("postal_address", ""))
    title = format_value(d.get("project_title", ""))
    return f"{org}, {addr}, Related works: {title}"


def transform_json_array(json_array):
    """
    Recursively transforms an array (possibly nested) of dictionaries into an array of strings.
    Each dictionary is converted using project_info_to_str.
    """
    result = []
    for item in json_array:
        if isinstance(item, list):
            result.extend(transform_json_array(item))
        elif isinstance(item, dict):
            result.append(project_info_to_str(item))
        else:
            result.append(str(item))
    return result


for entry in all_queries:
    for q_type, queries in entry.items():
        for query in queries:
            agent_result = asyncio.run(execute_agent_workflow(query))
            '''
            if q_type == "titles":
            vector_result = embedding_store.similarity_search_with_relevance_score(
                query_text=query,
                property_name="title",
                k=3
            )
            '''
            '''else:
                vector_result = embedding_store.similarity_search_with_relevance_score(
                    query_text=query,
                    property_name="abstract",
                    k=3
                )'''
            project_IRIs_by_similarity = embedding_store.similarity_search_with_relevance_score(
                query_text=query,
                property_name="abstract",
                k=3
            )

            project_IRIs = [item['iri'] for item in project_IRIs_by_similarity]

            vector_result = get_organisations_of_similar_projects(project_IRIs)

            final_prompt = f"""
            You are an expert providing information about european projects.
            Be as helpful as possible and return as much useful information as possible.
            Do not adding any additional useless information to the answer, for example do not answer that you are happy to help, do not answer you are sorry for the delay.
            Do not answer any questions that do not relate to projects, organisations, grants, fundings, or participants.
            Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.
        
            Answer only the question asked by the user. Do not provide any additional information that is not asked by the user.
        
            Your task is to analyze and synthesize information from two sources: the top result from a similarity search (unstructured information) and relevant data from a graph database (structured information).
            Given the user's query: {query}, provide a meaningful and efficient answer based on the insights derived from the following data:
        
            Unstructured information: {vector_result}.
            Structured information: {agent_result}
            """
            #res = llm.invoke(input=final_prompt)
            # print(res)
            #result = res.content
            # print(result)
            #results.append(result)
            results.append(agent_result)
            orgs = transform_json_array(vector_result)
            contexts.append(orgs)


def get_next_filename(base_filename: str, extension: str) -> str:
    """
    Determines the next available filename with a progressive count.

    Args:
        base_filename (str): The base filename without numbering.
        extension (str): The file extension (e.g., ".csv").

    Returns:
        str: The next available filename with an incremented number if necessary.
    """
    new_filename = f"{base_filename}{extension}"

    if os.path.exists(new_filename):
        existing_files = glob.glob(f"{base_filename}_*{extension}")
        highest_num = 0
        for file in existing_files:
            parts = file.replace(base_filename, "").replace(extension, "")
            if parts.startswith("_") and parts[1:].isdigit():
                highest_num = max(highest_num, int(parts[1:]))
        new_filename = f"{base_filename}_{highest_num + 1}{extension}"

    return new_filename


def eval_and_save_files(df: pd.DataFrame, base_filename: str, extension: str) -> str:
    """
    Saves the file with a progressive count if a file with the same name exists.
    Also computes means for numeric columns and saves them in a separate file.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        base_filename (str): The base filename without numbering.
        extension (str): The file extension (e.g., ".csv").

    Returns:
        str: The final saved filename.
    """
    new_filename = get_next_filename(base_filename, extension)

    # Save the DataFrame
    df.to_csv(new_filename, encoding="utf-8", index=False)
    print(f"Saved file as: {new_filename}")

    # Read the CSV file into a DataFrame
    df = pd.read_csv(new_filename)

    # Select only numeric columns
    numeric_df = df.select_dtypes(include="number")

    # Compute the mean for each numeric column and round to three decimals
    means = numeric_df.mean().round(3)

    # Create a DataFrame from the means (a single row)
    means_df = pd.DataFrame([means])

    # Generate progressive filename for means file
    means_filename = get_next_filename(base_filename + "_means", extension)

    # Save the means to a new CSV file with headers
    means_df.to_csv(means_filename, index=False)

    print(f"Saved means file as: {means_filename}")
    print("Computed means for numeric columns:")
    print(means_df)

    return new_filename


d = {
    "user_input": potential_consortium_queries,
    "retrieved_contexts": contexts,
    "response": results,
    "reference": potential_consortium_ground_truths,
}

# print("d: ", d)
dataset = Dataset.from_dict(d)
score = evaluate(dataset,
                 metrics=[faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall,
                          answer_similarity, answer_correctness],
                 llm=llm,
                 embeddings=embedder
                 )
score_df = score.to_pandas()

llm_name = "gpt-4o-20240513"
# llm_name = "claude-sonnet3.5-20241022"
base_filename = "evaluation/evaluation_scores_" + llm_name
extension = ".csv"

new_filename = eval_and_save_files(score_df, base_filename, extension)
print(f"Final saved filename: {new_filename}")
