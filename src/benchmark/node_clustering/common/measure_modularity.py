from networkx.algorithms.community import modularity


def measure_modularity_for(graph, embeddings, clustering_algorithm):
    clusters = []
    try:
        clusters = clustering_algorithm.fit(embeddings).labels_
        communities = []
        for _ in range(len(set(clusters))):
            communities.append(set())
        for idx, result in enumerate(clusters):
            communities[result].add(idx)

        return modularity(graph, communities, weight=None)
    except Exception as e:
        print("ERROR: Cannot measure modularity for clusters: %s" % clusters)
        return -1.55
