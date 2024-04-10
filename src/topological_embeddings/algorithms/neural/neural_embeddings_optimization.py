import torch
from torch import nn

from src.common.graph_utils import get_graph_shortest_paths_tensor
from src.common.torch_device import torch_device
from src.topological_embeddings.algorithms.common.rmrse_loss_function import rmrse_loss

default_hidden_layers = ((1024, 512), (512, 256), (256, 128))


def get_neural_embeddings(
        graph,
        embeddings_size,
        l_rate=0.001,
        n_epochs=300,
        loss_function=rmrse_loss,
        hidden_layers=default_hidden_layers,
        kappa=1.0
):
    graph_shortest_paths = get_graph_shortest_paths_tensor(graph).to(torch.float32)
    number_of_nodes = graph.number_of_nodes()
    if kappa is None:
        kappa_tensor = torch.rand(1).to(torch_device)
        kappa_tensor.requires_grad_()
        kappa_optimizer = torch.optim.Adam([kappa_tensor], lr=0.01)
    else:
        kappa_tensor = torch.tensor(kappa).to(torch_device)

    def train_node_classifier(model, optimizer):
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()

            embeddings = model(graph_shortest_paths)

            if kappa is None:
                kappa_optimizer.zero_grad()
                loss = loss_function(embeddings, graph_shortest_paths, torch.multiply(torch.sigmoid(kappa_tensor), 2.0))
            else:
                loss = loss_function(embeddings, graph_shortest_paths, kappa)

            loss.backward()
            optimizer.step()
            if kappa is None:
                kappa_optimizer.step()

        return [model, kappa_tensor]

    network = GraphEmbeddingForwardNeuralNetwork(hidden_layers, number_of_nodes, embeddings_size).to(torch_device)

    optimizer = torch.optim.Adam(params=network.parameters(), lr=l_rate, weight_decay=5e-4)
    [network, kappa_tensor] = train_node_classifier(network, optimizer)

    embeddings = network(graph_shortest_paths)
    if kappa is not None:
        return embeddings.cpu().detach().numpy(), torch.multiply(torch.sigmoid(kappa_tensor), 2.0).item()
    else:
        return [embeddings.cpu().detach().numpy(), kappa]


class GraphEmbeddingForwardNeuralNetwork(nn.Module):

    def __init__(self, hidden_layers_description, input_size, output_size):
        super().__init__()
        hidden_layers = list(map(lambda x: nn.Linear(x[0], x[1]), hidden_layers_description))
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_layers_description[0][0]),
            *hidden_layers,
            nn.Linear(hidden_layers_description[-1][1], output_size)
        )

    def forward(self, data):
        return self.layers(data)
