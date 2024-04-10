import math

import networkx as nx
import numpy as np
import torch

from src.common.torch_device import torch_device


def get_shortest_paths(graph):
    shortest_paths = dict(nx.shortest_path_length(graph))
    node_count = len(graph.nodes)
    anchor_descriptor = np.zeros((node_count, node_count))
    sorted_keys = sorted(shortest_paths.keys())
    for i in range(node_count):
        for j in range(node_count):
            if shortest_paths[sorted_keys[i]].get(sorted_keys[j]) is None:
                anchor_descriptor[i][j] = np.Infinity
            else:
                anchor_descriptor[i][j] = shortest_paths[sorted_keys[i]][sorted_keys[j]]
    return anchor_descriptor


def get_graph_shortest_paths_tensor(graph):
    shortest_paths = get_shortest_paths(graph)
    shortest_paths[shortest_paths == math.inf] = np.max(shortest_paths[shortest_paths != math.inf]) + 1
    return torch.from_numpy(shortest_paths).to(torch_device)
