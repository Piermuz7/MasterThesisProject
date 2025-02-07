from rag.embeddings import neo4j_embedding_service

from data_preparation.data_preparation import read_uris_from_csv, batch_delete_projects

#uris_to_delete = read_uris_from_csv("./data_preparation/project-uris-to-remove.csv")
#batch_delete_projects(uris_to_delete)

embedding_service.update_vector_indexes()