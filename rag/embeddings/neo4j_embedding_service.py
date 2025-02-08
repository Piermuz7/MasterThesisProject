import streamlit as st
import time
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector

embedder = OllamaEmbeddings(model="all-minilm:l6-v2")

def get_embedder():
    return embedder


def update_vector_indexes():
    """Updates vector indexes for title and abstract with real-time logging."""

    indexes = [
        {
            "index_name": "titleIndex",
            "node_label": "ns3__Project",
            "text_node_properties": ["ns3__title"],
            "embedding_node_property": "titleEmbedding"
        },
        {
            "index_name": "abstractIndex",
            "node_label": "ns3__Project",
            "text_node_properties": ["ns3__abstract"],
            "embedding_node_property": "abstractEmbedding"
        }
    ]

    for index in indexes:
        try:
            print(f"Indexing {index['text_node_properties'][0]}...")

            # Start timer
            start_time = time.time()

            # Create index
            Neo4jVector.from_existing_graph(
                embedder,
                url=st.secrets["NEO4J_URI"],
                username=st.secrets["NEO4J_USERNAME"],
                password=st.secrets["NEO4J_PASSWORD"],
                index_name=index["index_name"],
                node_label=index["node_label"],
                text_node_properties=index["text_node_properties"],
                embedding_node_property=index["embedding_node_property"],
            )

            # Measure time taken
            elapsed_time = time.time() - start_time
            print(f"Completed indexing {index['text_node_properties'][0]} in {elapsed_time:.2f} seconds.\n")

        except Exception as e:
            print(f"Error updating {index['text_node_properties'][0]} index: {e}")

    print("All indexes updated successfully!")
