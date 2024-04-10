from sklearn.cluster import MeanShift, DBSCAN, HDBSCAN, Birch, OPTICS, AffinityPropagation, AgglomerativeClustering

clustering_algorithms = [
    [MeanShift(), "MeanShift"],
    [MeanShift(bandwidth=0.25), "MeanShift 0.25"],
    [MeanShift(bandwidth=0.4), "MeanShift 0.4"],
    [MeanShift(bandwidth=0.5), "MeanShift 0.5"],
    [MeanShift(bandwidth=0.6), "MeanShift 0.6"],
    [MeanShift(bandwidth=0.75), "MeanShift 0.75"],
    [DBSCAN(min_samples=2), "DBSCAN"],
    [HDBSCAN(min_cluster_size=2), "HDBSCAN"],
    [Birch(n_clusters=None), "Birch"],
    [OPTICS(min_samples=2, metric="euclidean"), "OPTICS euclidean"],
    [OPTICS(min_samples=2), "OPTICS default"],
    [AffinityPropagation(damping=0.5), "AffinityPropagation 0.5"],
    [AffinityPropagation(damping=0.7), "AffinityPropagation 0.7"],
    [AffinityPropagation(damping=0.8), "AffinityPropagation 0.8"],
    [AffinityPropagation(damping=0.95), "AffinityPropagation 0.95"],
    [AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.5,
        compute_distances=True,
        compute_full_tree=True
    ), "AgglomerativeClustering 0.5"],
    [AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.8,
        compute_distances=True,
        compute_full_tree=True
    ), "AgglomerativeClustering 0.8"],
    [AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.95,
        compute_distances=True,
        compute_full_tree=True
    ), "AgglomerativeClustering 0.95"]
]
