from sklearn.utils.extmath import cartesian

from src.benchmark.node_clustering.common.benchmark_commons import clustering_algorithms
from src.benchmark.node_clustering.common.community_graphs.football_graph import FootballGraph
from src.benchmark.node_clustering.common.community_graphs.karate_club_graph import KarateClubGraph
from src.benchmark.node_clustering.true_labels_benchmark.true_labels_benchmark import run_true_labels_benchmark
from src.topological_embeddings.algorithms.common.rmrse_loss_function import rmrse_loss, count_rmrse_for
from src.topological_embeddings.algorithms.common.rmse_loss_function import count_rmse_for, rmse_loss
from src.topological_embeddings.algorithms.direct.linear_embeddings_optimization import get_direct_embeddings
from src.topological_embeddings.algorithms.neural.neural_embeddings_optimization import get_neural_embeddings

embedding_sizes = [2, 3, 5, 10, 15, 30, 50]

community_graphs = [
    KarateClubGraph(),
    FootballGraph(path="../common/community_graphs/data/football.gml")
]

main_path = "..."

loss_functions = [[rmse_loss, "RMSE", count_rmse_for], [rmrse_loss, "RMRSE", count_rmrse_for]]
embedding_algorithms = [[get_neural_embeddings, "neural"], [get_direct_embeddings, "direct"]]

for community_graph in community_graphs:
    for [[loss_fun, loss_fun_name, evaluate_fun], [embedding_algorithm, embedding_algorithm_name]] in cartesian(
            loss_functions, embedding_algorithms
    ):
        run_true_labels_benchmark(
            community_graph.get_graph(),
            embedding_algorithm=embedding_algorithm,
            clustering_algorithms=clustering_algorithms,
            output_csv_filename=r"%s\%s_%s_%s.csv" % (
                main_path, community_graph.get_name(), loss_fun_name, embedding_algorithm_name
            ),
            embedding_sizes=embedding_sizes,
            error_function=evaluate_fun,
            loss_function=loss_fun,
            true_labels=community_graph.get_true_labels()
        )
