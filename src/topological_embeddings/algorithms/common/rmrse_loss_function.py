import torch

from src.common.graph_utils import get_graph_shortest_paths_tensor
from src.common.torch_device import torch_device


def rmrse_loss(embeddings, graph_shortest_paths_tensor, kappa):
    distances = torch.cdist(embeddings, embeddings, p=2.0)
    distances = torch.pow(distances, kappa)

    mask = (graph_shortest_paths_tensor != 0)
    losses = torch.divide(distances[mask], graph_shortest_paths_tensor[mask])
    losses = torch.subtract(losses, 1)

    losses = torch.pow(losses, 2)
    losses = torch.sum(losses)
    losses = torch.divide(losses, distances.size(dim=1) * (distances.size(dim=1) - 1))

    return torch.sqrt(losses)


def count_rmrse_for(graph, embeddings, kappa):
    graph_shortest_paths_tensor = get_graph_shortest_paths_tensor(graph).to(torch_device)
    embeddings_tensor = torch.from_numpy(embeddings).to(torch_device)

    return rmrse_loss(embeddings_tensor, graph_shortest_paths_tensor, kappa=kappa).cpu().detach().item()
