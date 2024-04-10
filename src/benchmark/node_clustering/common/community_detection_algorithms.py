from networkx.algorithms.community import greedy_modularity_communities, louvain_communities, kernighan_lin_bisection

community_detection_algorithms = [
    [greedy_modularity_communities, "greedy_modularity_communities"],
    [louvain_communities, "louvain_communities"],
    [kernighan_lin_bisection, "kernighan_lin_bisection"],
    # [asyn_lpa_communities, "asyn_lpa_communities"]
]
