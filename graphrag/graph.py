import os

from langchain_community.graphs import OntotextGraphDBGraph
import paths as p
import streamlit as st

# Read from secrets.toml
GRAPHDB_ENDPOINT = st.secrets["GRAPHDB_URL"]

# Detect if running inside Docker
if os.getenv("DOCKER_ENV") == "true":
    GRAPHDB_ENDPOINT = GRAPHDB_ENDPOINT.replace("localhost", "graphdb")

# OntotextGraphDBGraph configuration
graph_db = OntotextGraphDBGraph(
    query_endpoint=GRAPHDB_ENDPOINT,
    local_file=p.ttl_absolute_path,
)
