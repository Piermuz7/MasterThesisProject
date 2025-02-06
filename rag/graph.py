from langchain_community.graphs import OntotextGraphDBGraph
import streamlit as st

graph = OntotextGraphDBGraph(
    query_endpoint=st.secrets["GRAPHDB_URL"],
    local_file='ontology/EURIO.ttl',
)