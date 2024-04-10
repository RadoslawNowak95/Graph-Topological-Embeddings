import torch

from src.common.graph_utils import get_graph_shortest_paths_tensor
from src.common.torch_device import torch_device


def rmse_loss(embeddings, graph_shortest_paths_tensor, kappa):
    distances = torch.cdist(embeddings, embeddings, p=2.0)
    distances = torch.pow(distances, kappa)

    losses = torch.subtract(distances, graph_shortest_paths_tensor)
    losses = torch.pow(losses, 2)
    losses = torch.sum(losses)
    losses = torch.divide(losses, distances.size(dim=1) * (distances.size(dim=1) - 1))

    return torch.sqrt(losses)


def count_rmse_for(graph, embeddings, kappa):
    graph_shortest_paths_tensor = get_graph_shortest_paths_tensor(graph).to(torch_device)
    embeddings_tensor = torch.from_numpy(embeddings).to(torch_device)

    return rmse_loss(embeddings_tensor, graph_shortest_paths_tensor, kappa=kappa).cpu().detach().item()
