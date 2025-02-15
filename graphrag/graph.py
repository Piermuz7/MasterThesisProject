from langchain_community.graphs import OntotextGraphDBGraph
import streamlit as st

# OntotextGraphDBGraph configuration
graph_db = OntotextGraphDBGraph(
    query_endpoint=st.secrets["GRAPHDB_URL"],
    local_file='ontology/EURIO.ttl',
)
