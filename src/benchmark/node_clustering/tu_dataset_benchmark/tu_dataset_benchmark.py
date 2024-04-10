import csv
import time

import numpy as np
from networkx.algorithms.community import modularity, louvain_communities, greedy_modularity_communities, girvan_newman, \
    kernighan_lin_bisection, asyn_lpa_communities
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

from src.benchmark.node_clustering.common.measure_modularity import measure_modularity_for


def tu_dataset_benchmark(
        dataset_name,
        embedding_algorithm,
        clustering_algorithms,
        output_csv_filename,
        embedding_sizes,
        error_function,
        loss_function,
        **loss_function_kwargs
):
    start_time = time.time()
    dataset = TUDataset(root='data/TUDataset', name=dataset_name)

    all_graphs = list(map(lambda dataset_graph: to_networkx(dataset_graph).to_undirected(), dataset))

    output_file = open(output_csv_filename, "a")
    cvs_writer = csv.writer(output_file, delimiter=';', quoting=csv.QUOTE_MINIMAL)
    cvs_writer.writerow(
        ["embeddings size", "error mean", "error median", "error std dev", "kappa mean", "kappa median",
         "kappa std dev", "clustering algorithm", "modularity mean", "modularity median", "modularity std dev"]
    )

    greedy_modularity_results = []
    louvain_results = []
    girvan_newman_results = []
    kernighan_lin_bisection_results = []
    asyn_lpa_communities_results = []
    for graph in all_graphs:
        greedy_modularity_results.append(
            modularity(graph, greedy_modularity_communities(graph), weight=None)
        )
        louvain_results.append(
            modularity(graph, louvain_communities(graph), weight=None)
        )
        girvan_newman_results.append(
            modularity(graph, next(girvan_newman(graph)), weight=None)
        )
        kernighan_lin_bisection_results.append(
            modularity(graph, kernighan_lin_bisection(graph), weight=None)
        )
        asyn_lpa_communities_results.append(
            modularity(graph, asyn_lpa_communities(graph), weight=None)
        )
    cvs_writer.writerow(
        ["-", "-", "-", "-", "-", "-", "-", "greedy_modularity_communities", np.mean(greedy_modularity_results),
         np.median(greedy_modularity_results), np.std(greedy_modularity_results)]
    )
    cvs_writer.writerow(
        ["-", "-", "-", "-", "-", "-", "-", "louvain_communities", np.mean(louvain_results), np.median(louvain_results),
         np.std(louvain_results)]
    )
    cvs_writer.writerow(
        ["-", "-", "-", "-", "-", "-", "-", "girvan_newman", np.mean(girvan_newman_results),
         np.median(girvan_newman_results), np.std(girvan_newman_results)]
    )
    cvs_writer.writerow(
        ["-", "-", "-", "-", "-", "-", "-", "kernighan_lin_bisection", np.mean(kernighan_lin_bisection_results),
         np.median(kernighan_lin_bisection_results), np.std(kernighan_lin_bisection_results)]
    )
    cvs_writer.writerow(
        ["-", "-", "-", "-", "-", "-", "-", "asyn_lpa_communities", np.mean(asyn_lpa_communities_results),
         np.median(asyn_lpa_communities_results), np.std(asyn_lpa_communities_results)]
    )
    output_file.flush()

    for embeddings_size in embedding_sizes:
        all_results = list(
            map(lambda g: embedding_algorithm(g, embeddings_size=embeddings_size, loss_function=loss_function,
                                              **loss_function_kwargs), all_graphs))
        all_kappa = list(map(lambda results: results[1], all_results))
        all_embeddings = list(map(lambda results: results[0], all_results))

        all_errors = list(
            map(lambda g, emb, kappa: error_function(g, emb, kappa=kappa), all_graphs, all_embeddings, all_kappa)
        )
        all_errors = list(filter(lambda e: e != -1.55, all_errors))

        mean_embeddings_error = np.mean(all_errors)
        median_embeddings_error = np.median(all_errors)
        std_dev_embeddings = np.std(all_errors)

        mean_kappa = np.mean(all_kappa)
        median_kappa = np.median(all_kappa)
        std_dev_kappa = np.std(all_kappa)

        for clustering_algorithm in clustering_algorithms:
            all_modularities = list(
                map(lambda g, emb: measure_modularity_for(g, emb, clustering_algorithm[0]),
                    all_graphs, all_embeddings
                    )
            )
            mean_modularity = np.mean(all_modularities)
            median_modularity = np.median(all_modularities)
            std_dev_modularity = np.std(all_modularities)

            cvs_writer.writerow(
                [embeddings_size, mean_embeddings_error, median_embeddings_error, std_dev_embeddings, mean_kappa,
                 median_kappa, std_dev_kappa, clustering_algorithm[1], mean_modularity, median_modularity,
                 std_dev_modularity]
            )
        output_file.flush()

    output_file.close()
    print("Completed: %s in %d s." % (dataset_name, time.time() - start_time))
