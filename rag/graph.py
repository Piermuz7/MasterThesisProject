from langchain_community.graphs import OntotextGraphDBGraph
from langchain_neo4j import Neo4jGraph
import streamlit as st

# OntotextGraphDBGraph configuration
graph_db = OntotextGraphDBGraph(
    query_endpoint=st.secrets["GRAPHDB_URL"],
    local_file='ontology/EURIO.ttl',
)

# Neo4jGraph configuration
neo4j_graph = Neo4jGraph(
    url=st.secrets["NEO4J_URI"],
    username=st.secrets["NEO4J_USERNAME"],
    password=st.secrets["NEO4J_PASSWORD"],
    enhanced_schema=True,
)