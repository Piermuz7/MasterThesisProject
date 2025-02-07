import time

from neo4j import GraphDatabase
from tqdm import tqdm  # Progress bar

import streamlit as st

CSV_FILE_PATH = "./project-uris-to-remove.csv"


# Read URIs from CSV
def read_uris_from_csv(file_path):
    with open(file_path, mode="r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


# Batch delete function with real-time logging
def batch_delete_projects(uri_list, batch_size=500):
    query = """
    UNWIND $uris AS uri
    MATCH (p:ns3__Project {uri: uri})
    DETACH DELETE p
    """

    total_nodes = len(uri_list)
    start_time = time.time()

    with GraphDatabase.driver(uri=st.secrets["NEO4J_URI"],
                              auth=(st.secrets["NEO4J_USERNAME"], st.secrets["NEO4J_PASSWORD"])) as driver:
        with driver.session() as session:
            with tqdm(total=total_nodes, desc="Deleting Projects", unit="proj") as pbar:
                for i in range(0, total_nodes, batch_size):
                    batch = uri_list[i: i + batch_size]
                    batch_start_time = time.time()

                    session.run(query, uris=batch)  # Run batch delete query

                    batch_time = time.time() - batch_start_time
                    processed = i + len(batch)
                    remaining = total_nodes - processed

                    # Estimated remaining time
                    avg_time_per_batch = (time.time() - start_time) / (i // batch_size + 1)
                    estimated_remaining_time = avg_time_per_batch * (remaining // batch_size)

                    # Logging
                    print(f"Deleted {len(batch)} projects | Processed: {processed}/{total_nodes} | "
                          f"Batch time: {batch_time:.2f}s | Est. remaining: {estimated_remaining_time:.2f}s")

                    pbar.update(len(batch))  # Update progress bar

    print(f"All {total_nodes} projects deleted in {time.time() - start_time:.2f} seconds.")
