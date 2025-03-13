from graphrag.embeddings.graph_embedding_service import GraphEmbeddingStore
from SPARQLWrapper import SPARQLWrapper, JSON
import streamlit as st

# Initialize embedding store
embedding_store = GraphEmbeddingStore()


# Function to run SPARQL query and get projects with abstracts
def fetch_projects_from_graph():
    sparql = SPARQLWrapper(st.secrets["GRAPHDB_URL"])
    sparql.setReturnFormat(JSON)

    query = """
    PREFIX eurio: <http://data.europa.eu/s66#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?project ?title ?abstract
    WHERE {
        ?project a eurio:Project .
        ?project eurio:title ?title.
        ?project eurio:abstract ?abstract.

        FILTER EXISTS {
            ?project eurio:hasInvolvedParty ?party .
    	    ?party a eurio:OrganisationRole.
    	    ?party eurio:isRoleOf ?role .
    	    ?person eurio:isInvolvedIn ?project.
		    ?person eurio:isEmployedBy ?role .
    	    ?person eurio:isRoleOf ?p.
        }
    }
    """

    sparql.setQuery(query)
    response = sparql.query().convert()

    projects = []
    for result in response["results"]["bindings"]:
        iri = result["project"]["value"]
        abstract = result["abstract"]["value"]
        title = result["title"]["value"]
        projects.append((iri, title, abstract))

    return projects


# print(fetch_projects_from_graph()[0])


# Fetch projects from GraphDB
projects = fetch_projects_from_graph()

'''
projects = [
    ("http://data.europa.eu/s66/resource/projects/9cf329d2-f10f-37fc-8743-fff766dcf1ca",
     "BIM-based holistic tools for Energy-driven Renovation of existing Residences",
     "This project aims to develop BIM-based holistic tools for Energy-driven Renovation of existing Residences.")
]
'''


def batch_store_embeddings(embedding_store, projects, batch_size=1000):
    """Store embeddings in batches to optimize processing."""
    total_projects = len(projects)
    num_batches = (total_projects // batch_size) + (1 if total_projects % batch_size != 0 else 0)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_projects)
        batch = projects[start_idx:end_idx]

        print(f"Processing batch {i + 1}/{num_batches} (Size: {len(batch)})")
        embedding_store.store_embeddings(batch)

    print("All projects have been processed and stored successfully.")


batch_store_embeddings(embedding_store, projects, batch_size=1000)

# Store embeddings
embedding_store.store_embeddings(projects)

#print("Total Embeddings Stored:", len(embedding_store.collection.get()["ids"]))

# Test retrieval
# test_iri = "http://data.europa.eu/s66/resource/projects/9cf329d2-f10f-37fc-8743-fff766dcf1ca"
# title_embedding = embedding_store.get_embedding(test_iri, "title")
# abstract_embedding = embedding_store.get_embedding(test_iri, "abstract")

# print("Retrieved title embedding:", title_embedding)
# print("Retrieved abstract embedding:", abstract_embedding)

# Test similarity search
# query_text = "Topology driven methods project"
# similar_entities = embedding_store.similarity_search_with_relevance_score(query_text, "title", k=3)
# print("Similar entities:", similar_entities)
