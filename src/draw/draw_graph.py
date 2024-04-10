from matplotlib import pyplot as plt
from networkx import all_neighbors

from src.topological_embeddings.algorithms.common.rmrse_loss_function import rmrse_loss
from src.topological_embeddings.algorithms.direct.linear_embeddings_optimization import get_direct_embeddings


def draw_graph(
        graph,
        algorithm=get_direct_embeddings,
        loss_function=rmrse_loss,
        kappa=None,
        connect_nodes=False,
        annotate=True,
        save_filename=None,
        add_info=False,
        add_info_error_name=None,
        algorithm_info_name=None,
        error_function=None,
        title=None,
):
    [embeddings, final_kappa] = algorithm(graph=graph, embeddings_size=2, loss_function=loss_function, kappa=kappa)
    fig = plt.figure()
    x = embeddings[:, 0]
    y = embeddings[:, 1]
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y)

    if annotate:
        for i in range(0, len(x)):
            ax.annotate(str(i), (x[i], y[i]))

    if add_info:
        error = error_function(graph, embeddings, kappa=final_kappa)
        ax.set_xlabel(
            "Algorithm: %s; kappa=%.2f. %s=%.2f" % (algorithm_info_name, final_kappa, add_info_error_name, error),
            fontsize=7
        )

    if title is not None:
        ax.set_title(title)

    if connect_nodes:
        for node in graph.nodes:
            for neighbour in all_neighbors(graph, node):
                plt.plot(
                    [embeddings[node][0], embeddings[neighbour][0]],
                    [embeddings[node][1], embeddings[neighbour][1]],
                    linewidth=0.25,
                    color="gray"
                )

    if save_filename is None:
        plt.show()
    else:
        plt.savefig(save_filename)
