# future work

from .base import BaseDBOperations

class Neo4jOperations(BaseDBOperations):
    def __init__(self, uri=None, user=None, password=None):
        self.driver = None
        if uri and user and password:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        if self.driver:
            self.driver.close()

    def insert_node(self, label, properties):
        if not self.driver:
            print("Neo4j driver not initialized")
            return
        with self.driver.session() as session:
            session.write_transaction(self._create_node, label, properties)

    @staticmethod
    def _create_node(tx, label, properties):
        query = f"CREATE (n:{label} {{ {', '.join([f'{k}: ${k}' for k in properties.keys()])} }})"
        tx.run(query, **properties)

    def update_node(self, label, identifier, updates):
        if not self.driver:
            print("Neo4j driver not initialized")
            return
        with self.driver.session() as session:
            session.write_transaction(self._update_node, label, identifier, updates)

    @staticmethod
    def _update_node(tx, label, identifier, updates):
        query = f"""
        MATCH (n:{label} {{id: $id}})
        SET {', '.join([f'n.{k} = ${k}' for k in updates.keys()])}
        """
        tx.run(query, id=identifier, **updates)

    def delete_node(self, label, identifier):
        if not self.driver:
            print("Neo4j driver not initialized")
            return
        with self.driver.session() as session:
            session.write_transaction(self._delete_node, label, identifier)

    @staticmethod
    def _delete_node(tx, label, identifier):
        query = f"MATCH (n:{label} {{id: $id}}) DELETE n"
        tx.run(query, id=identifier)