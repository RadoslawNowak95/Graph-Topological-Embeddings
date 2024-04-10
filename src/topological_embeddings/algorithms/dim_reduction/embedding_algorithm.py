from abc import ABC, abstractmethod

import numpy as np

from src.common.graph_utils import get_shortest_paths


class EmbeddingAlgorithm(ABC):

    def __init__(self):
        self.graph = None
        self.nodes_count = None
        self.embedding_size = None
        self.graph_shortest_paths = None

    def calculate_embeddings_for(
            self,
            graph,
            embedding_size,
            get_shortest_paths_alg=get_shortest_paths
    ):
        if graph.number_of_nodes() == 1:
            return [np.zeros(embedding_size)]

        self.graph = graph
        self.nodes_count = graph.number_of_nodes()
        self.embedding_size = embedding_size
        self.graph_shortest_paths = get_shortest_paths_alg(graph)

        embeddings = self.calculate_embeddings()
        return embeddings

    @abstractmethod
    def calculate_embeddings(self):
        pass

    @abstractmethod
    def get_description(self):
        pass
