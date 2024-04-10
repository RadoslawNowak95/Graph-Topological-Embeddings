import networkx as nx

from src.benchmark.node_clustering.common.community_graphs.community_graph import CommunityGraph


class FootballGraph(CommunityGraph):

    def get_name(self):
        return "Football"

    def __init__(self, path):
        self.graph = nx.read_gml(path, label=None)

    def get_number_of_communities(self):
        return 12

    def get_graph(self):
        return self.graph

    def get_true_labels(self):
        labels = []
        for idx in self.graph.nodes:
            labels.append(self.graph.nodes[idx]["value"])
        return labels
