import chromadb
from chromadb.utils import embedding_functions
import logging
import re
from SPARQLWrapper import SPARQLWrapper, JSON
import streamlit as st
from typing import List, Tuple, Optional, Dict, Any


class GraphEmbeddingStore:
    def __init__(self, persist_directory="./chroma_db"):
        """Initialize the embedding store with ChromaDB."""
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        # Create collections if they don't exist
        self.title_collection = self.client.get_or_create_collection(
            name="title_embeddings",
            embedding_function=self.embedding_function,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:search_ef": 100
            }
        )
        self.abstract_collection = self.client.get_or_create_collection(
            name="abstract_embeddings",
            embedding_function=self.embedding_function,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:search_ef": 100
            }
        )

        self.collections = {
            "title": self.title_collection,
            "abstract": self.abstract_collection
        }

    def generate_embedding(self, text):
        """Generate embeddings for the given text."""
        if not text or not isinstance(text, str):
            logging.warning(f"Invalid text for embedding: {text}")
            return None
        return self.embedding_function([text])[0]

    def store_embeddings(self, entity_data: List[Tuple[str, Optional[str], Optional[str]]]):
        """Store embeddings for multiple entities in batch with proper handling of duplicates.

        Parameters:
        - entity_data: List of tuples (iri, title, abstract) where title and abstract can be None
        """
        # First, collect all IDs we might need to process
        all_title_ids = []
        all_abstract_ids = []

        processed_data: Dict[str, Dict[str, Any]] = {}

        for iri, title, abstract in entity_data:
            if not iri:
                logging.warning(f"Skipping entity without IRI: {(iri, title, abstract)}")
                continue

            # Generate a safe ID from the IRI
            safe_id = iri.replace('/', '_').replace(':', '_')
            title_id = f"title_{safe_id}"
            abstract_id = f"abstract_{safe_id}"

            processed_data[iri] = {
                "title": title,
                "abstract": abstract,
                "title_id": title_id,
                "abstract_id": abstract_id
            }

            if title:
                all_title_ids.append(title_id)
            if abstract:
                all_abstract_ids.append(abstract_id)

        # Batch check which IDs already exist in collections
        existing_title_ids = set()
        existing_abstract_ids = set()

        if all_title_ids:
            try:
                existing_title_results = self.title_collection.get(
                    ids=all_title_ids,
                    include=[]  # We only need to know if they exist
                )
                existing_title_ids = set(existing_title_results["ids"])
            except Exception as e:
                # Handle case where none of the IDs exist
                logging.debug(f"Error checking title IDs: {e}")
                print(f"Error checking title IDs: {e}")

        if all_abstract_ids:
            try:
                existing_abstract_results = self.abstract_collection.get(
                    ids=all_abstract_ids,
                    include=[]  # We only need to know if they exist
                )
                existing_abstract_ids = set(existing_abstract_results["ids"])
            except Exception as e:
                # Handle case where none of the IDs exist
                logging.debug(f"Error checking abstract IDs: {e}")
                print(f"Error checking abstract IDs: {e}")

        # Prepare batches for add and update operations
        new_title_docs = []
        new_title_metadatas = []
        new_title_ids = []

        new_abstract_docs = []
        new_abstract_metadatas = []
        new_abstract_ids = []

        # For updates, we need to handle them one by one
        for iri, data in processed_data.items():
            title = data.get("title")
            abstract = data.get("abstract")
            title_id = data.get("title_id")
            abstract_id = data.get("abstract_id")

            if title:
                if title_id in existing_title_ids:
                    # Update existing title
                    try:
                        self.title_collection.update(
                            ids=[title_id],
                            documents=[title],
                            metadatas=[{"iri": iri}]
                        )
                        logging.info(f"Updated existing title for IRI: {iri}")
                        print(f"Updated existing title for IRI: {iri}")
                    except Exception as e:
                        logging.error(f"Error updating title for IRI {iri}: {e}")
                        print(f"Error updating title for IRI {iri}: {e}")
                else:
                    # Add to batch for new titles
                    new_title_docs.append(title)
                    new_title_metadatas.append({"iri": iri})
                    new_title_ids.append(title_id)

            if abstract:
                if abstract_id in existing_abstract_ids:
                    # Update existing abstract
                    try:
                        self.abstract_collection.update(
                            ids=[abstract_id],
                            documents=[abstract],
                            metadatas=[{"iri": iri}]
                        )
                        logging.info(f"Updated existing abstract for IRI: {iri}")
                        print(f"Updated existing abstract for IRI: {iri}")
                    except Exception as e:
                        logging.error(f"Error updating abstract for IRI {iri}: {e}")
                        print(f"Error updating abstract for IRI {iri}: {e}")
                else:
                    # Add to batch for new abstracts
                    new_abstract_docs.append(abstract)
                    new_abstract_metadatas.append({"iri": iri})
                    new_abstract_ids.append(abstract_id)

        # Add new titles in batch
        if new_title_docs:
            self.title_collection.add(
                documents=new_title_docs,
                metadatas=new_title_metadatas,
                ids=new_title_ids
            )
            logging.info(f"Added {len(new_title_docs)} new title embeddings")
            print(f"Added {len(new_title_docs)} new title embeddings")

        # Add new abstracts in batch
        if new_abstract_docs:
            self.abstract_collection.add(
                documents=new_abstract_docs,
                metadatas=new_abstract_metadatas,
                ids=new_abstract_ids
            )
            logging.info(f"Added {len(new_abstract_docs)} new abstract embeddings")
            print(f"Added {len(new_abstract_docs)} new abstract embeddings")

    def similarity_search_with_relevance_score(self, query_text, property_name, k=5):
        """Find entities with similar embeddings for a specific property and return their merged data properties with relevance scores."""
        if property_name not in self.collections:
            raise ValueError(f"Invalid property name: {property_name}. Must be 'title' or 'abstract'")

        collection = self.collections[property_name]

        # Query the collection
        results = collection.query(
            query_texts=[query_text],
            n_results=k
        )

        if not results["metadatas"] or len(results["metadatas"][0]) == 0:
            return []

        entity_results = []

        # Process each result
        for i, metadata in enumerate(results["metadatas"][0]):
            iri = metadata["iri"]
            document = results["documents"][0][i]
            distance = results["distances"][0][i] if "distances" in results else None

            # Convert distance to similarity score (assuming cosine distance)
            # ChromaDB typically returns distance, so we convert to similarity
            relevance_score = 1 - distance if distance is not None else None

            # Get all properties from GraphDB
            properties = self.__get_data_properties_from_graph(iri)

            # Add the matching property and relevance score
            result_item = {
                "iri": iri,
                property_name: document,
                "relevance_score": relevance_score,
                **properties
            }

            entity_results.append(result_item)

        return entity_results

    def get_entity_by_iri(self, iri):
        """Retrieve entity data by exact IRI match."""
        properties = self.__get_data_properties_from_graph(iri)

        # Generate safe ID for lookups
        safe_id = iri.replace('/', '_').replace(':', '_')
        title_id = f"title_{safe_id}"
        abstract_id = f"abstract_{safe_id}"

        # Try to get the title and abstract if available
        try:
            title_results = self.title_collection.get(ids=[title_id])
            if title_results["ids"]:
                properties["title"] = title_results["documents"][0]
        except Exception:
            logging.debug(f"No title found for IRI: {iri}")

        try:
            abstract_results = self.abstract_collection.get(ids=[abstract_id])
            if abstract_results["ids"]:
                properties["abstract"] = abstract_results["documents"][0]
        except Exception:
            logging.debug(f"No abstract found for IRI: {iri}")

        # Combine all data
        result = {"iri": iri, **properties}
        return result

    def __get_data_properties_from_graph(self, iri: str):
        """Retrieve data properties from the graph for a given IRI and return a clean JSON object."""
        sparql = SPARQLWrapper(st.secrets["GRAPHDB_URL"])
        sparql.setReturnFormat(JSON)

        query = f"""
            PREFIX eurio: <http://data.europa.eu/s66#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT * WHERE {{
                <{iri}> ?p ?o .
                FILTER(isLiteral(?o))
            }}
            """

        sparql.setQuery(query)
        response = sparql.query().convert()

        # Process results into a clean dictionary
        properties = {}

        for result in response["results"]["bindings"]:
            property_uri = result["p"]["value"]
            value = result["o"]["value"]

            # Extract the last part of the URI as the property name
            property_name = re.sub(r".*[/#]", "", property_uri)  # Removes namespace

            # Handle multiple values for the same property
            if property_name in properties:
                if isinstance(properties[property_name], list):
                    properties[property_name].append(value)
                else:
                    properties[property_name] = [properties[property_name], value]
            else:
                properties[property_name] = value

        logging.info("Retrieved %d data properties for IRI: %s", len(properties), iri)
        return properties
