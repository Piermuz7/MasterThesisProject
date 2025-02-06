from pymongo import MongoClient
import numpy as np
from sentence_transformers import SentenceTransformer

import streamlit as st



class EmbeddingStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize MongoDB connection and Sentence Transformer model."""
        self.client = MongoClient(st.secrets["MONGO_URI"])
        self.db = self.client[st.secrets["DB_NAME"]]
        self.collection = self.db[st.secrets["COLLECTION_NAME"]]
        self.model = SentenceTransformer(model_name)

    def __generate_embedding(self, text):
        """Generate an embedding for the given text."""
        return self.model.encode(text).tolist()

    def add_embedding(self, iri, property_name, text):
        """Add or update an embedding for a specific entity and property."""
        embedding = self.__generate_embedding(text)

        # Update or insert the new embedding
        self.collection.update_one(
            {"_id": iri},
            {"$set": {f"embeddings.{property_name}": embedding}},
            upsert=True
        )
        return f"Embedding for {property_name} added/updated for {iri}."

    def remove_embedding(self, iri, property_name):
        """Remove a specific embedding for a given entity and property."""
        update_result = self.collection.update_one(
            {"_id": iri},
            {"$unset": {f"embeddings.{property_name}": ""}}
        )
        if update_result.modified_count > 0:
            return f"Embedding for {property_name} removed for {iri}."
        return f"No embedding found for {property_name} on {iri}."

    def get_embedding(self, iri, property_name):
        """Retrieve an embedding for a specific entity and property."""
        result = self.collection.find_one({"_id": iri}, {"embeddings": 1})
        if result and "embeddings" in result and property_name in result["embeddings"]:
            return np.array(result["embeddings"][property_name])
        return None

    def find_similar_entities(self, query_text, property_name, k=5):
        """Find entities with similar embeddings for a specific property."""
        query_embedding = self.__generate_embedding(query_text)
        all_entities = self.collection.find({}, {"_id": 1, f"embeddings.{property_name}": 1})

        similarities = []
        for entity in all_entities:
            if "embeddings" in entity and property_name in entity["embeddings"]:
                stored_embedding = np.array(entity["embeddings"][property_name])
                similarity = np.dot(query_embedding, stored_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding))
                similarities.append((entity["_id"], similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
