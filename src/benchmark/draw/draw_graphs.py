from networkx import read_weighted_edgelist
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

from src.draw.draw_graph import draw_graph
from src.topological_embeddings.algorithms.common.rmrse_loss_function import count_rmrse_for, rmrse_loss
from src.topological_embeddings.algorithms.common.rmse_loss_function import count_rmse_for, rmse_loss
from src.topological_embeddings.algorithms.direct.linear_embeddings_optimization import get_direct_embeddings
from src.topological_embeddings.algorithms.linear_transform.linear_transform_embeddings_optimization import \
    get_linear_transform_embeddings
from src.topological_embeddings.algorithms.neural.neural_embeddings_optimization import get_neural_embeddings


def get_tu_graph(name, idx=100):
    dataset = TUDataset(root='data/TUDataset', name=name)
    return to_networkx(dataset[idx]).to_undirected()


graphs = [
    [
        "dolphins",
        read_weighted_edgelist(path="../node_clustering/common/community_graphs/data/dolphins.txt", nodetype=int)
    ],
    [
        "les_miserables",
        read_weighted_edgelist(path="../node_clustering/common/community_graphs/data/les_miserables.txt",
                               nodetype=int)
    ],
    [
        "political_books",
        read_weighted_edgelist(path="../node_clustering/common/community_graphs/data/political_books.txt",
                               nodetype=int)
    ],
    [
        "train_bombing",
        read_weighted_edgelist(path="../node_clustering/common/community_graphs/data/train_bombing.txt",
                               nodetype=int)
    ],
    [
        "tribes",
        read_weighted_edgelist(path="../node_clustering/common/community_graphs/data/tribes.txt", nodetype=int)
    ],
    [
        "windsurfers",
        read_weighted_edgelist(path="../node_clustering/common/community_graphs/data/windsurfers.txt", nodetype=int)
    ],
    [
        "zebra",
        read_weighted_edgelist(path="../node_clustering/common/community_graphs/data/zebra.txt", nodetype=int)
    ],
    [
        "revolution",
        read_weighted_edgelist(path="../node_clustering/common/community_graphs/data/revolution.txt", nodetype=int)
    ],
    ["mutag", get_tu_graph("MUTAG")],
    ["enzymes", get_tu_graph("ENZYMES")],
    ["proteins", get_tu_graph("PROTEINS")],
    ["reddit_binary", get_tu_graph("REDDIT-BINARY")],
    ["collab", get_tu_graph("COLLAB")],
]

base_draw_dir = r"/home/radekpriv/draw"
kappas = [None, 1.0, 1.5]

for kappa in kappas:
    for graph in graphs:
        draw_graph(
            graph[1],
            algorithm=get_direct_embeddings,
            connect_nodes=True,
            save_filename=r"%s/kappa%s/%s_direct_rmse.png" % (base_draw_dir, kappa, graph[0]),
            add_info=True,
            add_info_error_name="RMSE",
            algorithm_info_name="Direct",
            error_function=count_rmse_for,
            loss_function=rmse_loss,
            title=graph[0],
            kappa=kappa
        )

        draw_graph(
            graph[1],
            algorithm=get_direct_embeddings,
            connect_nodes=True,
            save_filename=r"%s/kappa%s/%s_direct_rmrse.png" % (base_draw_dir, kappa, graph[0]),
            add_info=True,
            add_info_error_name="RMRSE",
            algorithm_info_name="Direct",
            error_function=count_rmrse_for,
            loss_function=rmrse_loss,
            title=graph[0],
            kappa=kappa
        )

        draw_graph(
            graph[1],
            algorithm=get_linear_transform_embeddings,
            connect_nodes=True,
            save_filename=r"%s/kappa%s/%s_linear_transform_rmse.png" % (base_draw_dir, kappa, graph[0]),
            add_info=True,
            add_info_error_name="RMSE",
            algorithm_info_name="Linear transform",
            error_function=count_rmse_for,
            loss_function=rmse_loss,
            title=graph[0],
            kappa=kappa
        )

        draw_graph(
            graph[1],
            algorithm=get_linear_transform_embeddings,
            connect_nodes=True,
            save_filename=r"%s/kappa%s/%s_linear_transform_rmrse.png" % (base_draw_dir, kappa, graph[0]),
            add_info=True,
            add_info_error_name="RMRSE",
            algorithm_info_name="Linear transform",
            error_function=count_rmrse_for,
            loss_function=rmrse_loss,
            title=graph[0],
            kappa=kappa
        )

        draw_graph(
            graph[1],
            algorithm=get_neural_embeddings,
            connect_nodes=True,
            save_filename=r"%s/kappa%s/%s_neural_rmse.png" % (base_draw_dir, kappa, graph[0]),
            add_info=True,
            add_info_error_name="RMSE",
            algorithm_info_name="Neural",
            error_function=count_rmse_for,
            loss_function=rmse_loss,
            title=graph[0],
            kappa=kappa
        )

        draw_graph(
            graph[1],
            algorithm=get_neural_embeddings,
            connect_nodes=True,
            save_filename=r"%s/kappa%s/%sneural_rmrse.png" % (base_draw_dir, kappa, graph[0]),
            add_info=True,
            add_info_error_name="RMRSE",
            algorithm_info_name="Neural",
            error_function=count_rmrse_for,
            loss_function=rmrse_loss,
            title=graph[0],
            kappa=kappa
        )
