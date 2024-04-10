import math
from abc import ABC

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import locally_linear_embedding, Isomap

from src.common.graph_utils import get_shortest_paths
from src.topological_embeddings.algorithms.dim_reduction.embedding_algorithm import EmbeddingAlgorithm


class DimReductionAlgorithm(EmbeddingAlgorithm, ABC):

    def calculate_embeddings_for(
            self,
            graph,
            embedding_size,
            get_shortest_paths_alg=get_shortest_paths
    ):
        number_of_nodes = graph.number_of_nodes()
        if number_of_nodes == 1:
            return np.zeros((number_of_nodes, embedding_size))
        elif embedding_size > number_of_nodes:
            embeddings = super().calculate_embeddings_for(graph=graph, embedding_size=number_of_nodes)
            return np.hstack((
                embeddings, np.zeros((number_of_nodes, embedding_size - number_of_nodes))
            ))
        else:
            return super().calculate_embeddings_for(graph=graph, embedding_size=embedding_size)


class PcaEmbeddingAlgorithm(DimReductionAlgorithm):

    def calculate_embeddings(self):
        pca = PCA(n_components=self.embedding_size)
        return pca.fit_transform(self.graph_shortest_paths)

    def get_description(self):
        return "PCA"


class KPcaEmbeddingAlgorithm(DimReductionAlgorithm):

    def __init__(self, kernel="rbf"):
        super().__init__()
        self.kernel = kernel

    def calculate_embeddings(self):
        k_pca = KernelPCA(n_components=self.embedding_size, kernel=self.kernel)
        return k_pca.fit_transform(self.graph_shortest_paths)

    def get_description(self):
        return "KPCA___ker=%s" % self.kernel


class LleEmbeddingAlgorithm(DimReductionAlgorithm):

    def __init__(self, max_iter=100, n_neighbours_fraction=1.0):
        super().__init__()
        self.max_iter = max_iter
        self.n_neighbours_fraction = n_neighbours_fraction

    def calculate_embeddings(self):
        n_neighbors = math.ceil(self.nodes_count * self.n_neighbours_fraction)
        if n_neighbors >= self.nodes_count:
            n_neighbors = self.nodes_count - 1

        result = locally_linear_embedding(
            X=self.graph_shortest_paths,
            n_neighbors=n_neighbors,
            n_components=self.embedding_size,
            eigen_solver='dense',
            max_iter=self.max_iter,
            n_jobs=-1
        )
        return result[0]

    def get_description(self):
        return "LLE___max_iter=%d-n_neighbours_fraction=%.2f" % (self.max_iter, self.n_neighbours_fraction)


class IsomapEmbeddingAlgorithm(DimReductionAlgorithm):

    def calculate_embeddings(self):
        return Isomap(
            n_components=self.embedding_size,
            n_neighbors=self.nodes_count - 1,
            eigen_solver="auto"
        ).fit_transform(self.graph_shortest_paths)

    def get_description(self):
        return "IsoMap"
