from abc import ABC, abstractmethod


class BaseDBOperations(ABC):
    @abstractmethod
    def insert_node(self, label, properties):
        pass

    @abstractmethod
    def update_node(self, label, identifier, updates):
        pass

    @abstractmethod
    def delete_node(self, label, identifier):
        pass
