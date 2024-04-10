import numpy as np
import torch

from src.common.graph_utils import get_graph_shortest_paths_tensor
from src.common.torch_device import torch_device
from src.topological_embeddings.algorithms.common.rmrse_loss_function import rmrse_loss


def get_linear_transform_embeddings(graph, embeddings_size, l_rate=0.01, iterations=1000, loss_function=rmrse_loss,
                                    kappa=1.0):
    graph_shortest_paths = get_graph_shortest_paths_tensor(graph)
    number_of_nodes = graph.number_of_nodes()
    if kappa is None:
        kappa_tensor = torch.rand(1).to(torch_device)
        kappa_tensor.requires_grad_()
        kappa_optimizer = torch.optim.Adam([kappa_tensor], lr=0.01)
    else:
        kappa_tensor = torch.tensor(kappa).to(torch_device)

    def optimize_embeddings():
        embeddings_transform_matrix = torch.rand(number_of_nodes, embeddings_size).to(torch_device).to(torch.float64)
        embeddings_transform_matrix.requires_grad_()
        optimizer = torch.optim.Adam([embeddings_transform_matrix], lr=l_rate)

        for i in range(iterations):
            optimizer.zero_grad()
            embeddings = torch.matmul(graph_shortest_paths, embeddings_transform_matrix)
            if kappa is None:
                kappa_optimizer.zero_grad()
                loss = loss_function(embeddings, graph_shortest_paths, torch.multiply(torch.sigmoid(kappa_tensor), 2.0))
            else:
                loss = loss_function(embeddings, graph_shortest_paths, kappa)
            loss.backward()
            optimizer.step()
            if kappa is None:
                kappa_optimizer.step()

        embeddings = torch.matmul(graph_shortest_paths, embeddings_transform_matrix)
        if kappa is None:
            return [embeddings.cpu().detach().numpy(), torch.multiply(torch.sigmoid(kappa_tensor), 2.0).item()]
        else:
            return [embeddings.cpu().detach().numpy(), kappa]

    if number_of_nodes == 1:
        return [np.zeros(embeddings_size)]
    else:
        return optimize_embeddings()
