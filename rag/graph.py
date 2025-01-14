import os

import streamlit as st
from langchain_community.graphs import OntotextGraphDBGraph

# GraphDB configuration
graph = OntotextGraphDBGraph(
    query_endpoint=st.secrets["GRAPHDB_URL"],
    local_file='../ontology/EURIO.ttl'
)
