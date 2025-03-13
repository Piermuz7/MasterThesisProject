import streamlit as st
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ttl_absolute_path = str(os.path.join(PROJECT_ROOT, st.secrets["EURIO_ONTOLOGY_PATH"]))