import csv

from networkx.algorithms.community import modularity
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, \
    v_measure_score

from src.benchmark.node_clustering.common.community_detection_algorithms import community_detection_algorithms


def get_scores(true_labels, actual_labels, graph, clusters):
    ars = adjusted_rand_score(true_labels, actual_labels)
    nmi = normalized_mutual_info_score(true_labels, actual_labels)
    homo = homogeneity_score(true_labels, actual_labels)
    comp = completeness_score(true_labels, actual_labels)
    vmes = v_measure_score(true_labels, actual_labels)
    mod = modularity(graph, clusters, weight=None)

    return [ars, nmi, homo, comp, vmes, mod]


def run_true_labels_benchmark(
        graph,
        embedding_algorithm,
        clustering_algorithms,
        output_csv_filename,
        embedding_sizes,
        error_function,
        loss_function,
        true_labels
):
    output_file = open(output_csv_filename, "a")
    cvs_writer = csv.writer(output_file, delimiter=';', quoting=csv.QUOTE_MINIMAL)
    cvs_writer.writerow(
        ["clustering alg", "embedding size", "error", "modularity", "ars", "nmi", "homo", "comp", "v-mes"]
    )

    for algorithm in community_detection_algorithms:
        clusters = algorithm[0](graph)

        labels = [0] * len(graph.nodes)
        for idx, result in enumerate(clusters):
            for single_result in graph.subgraph(result):
                labels[single_result] = idx

        ars, nmi, homo, comp, vmes, mod = get_scores(true_labels, labels, graph, clusters)

        cvs_writer.writerow([algorithm[1], "-", "-", mod, ars, nmi, homo, comp, vmes])

    for embeddings_size in embedding_sizes:

        embeddings = embedding_algorithm(graph, embeddings_size, loss_function=loss_function, auto_kappa=True)
        for [clustering_algorithm, clustering_algorithm_name] in clustering_algorithms:
            labels = clustering_algorithm.fit(embeddings).labels_

            clusters = []
            for _ in range(len(set(labels))):
                clusters.append(set())
            for idx, result in enumerate(labels):
                clusters[result].add(idx)

            ars, nmi, homo, comp, vmes, mod = get_scores(true_labels, labels, graph, clusters)
            error = error_function(graph, embeddings)

            cvs_writer.writerow([clustering_algorithm_name, embeddings_size, error, mod, ars, nmi, homo, comp, vmes])

    output_file.close()
