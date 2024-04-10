import networkx as nx

from src.benchmark.node_clustering.common.community_graphs.community_graph import CommunityGraph


class KarateClubGraph(CommunityGraph):

    def get_name(self):
        return "ZacharyKarateClub"

    def __init__(self):
        self.graph = nx.karate_club_graph()

    def get_number_of_communities(self):
        return 2

    def get_graph(self):
        return self.graph

    def get_true_labels(self):
        labels = []
        for idx in range(len(self.graph.nodes)):
            labels.append(self.graph.nodes[idx]["club"])
        return [0 if x == "Mr. Hi" else 1 for x in labels]
