from abc import ABC, abstractmethod


class CommunityGraph(ABC):

    @abstractmethod
    def get_number_of_communities(self):
        pass

    @abstractmethod
    def get_graph(self):
        pass

    @abstractmethod
    def get_true_labels(self):
        pass

    @abstractmethod
    def get_name(self):
        pass
