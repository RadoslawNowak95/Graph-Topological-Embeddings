from sklearn.utils.extmath import cartesian

from src.benchmark.node_clustering.common.benchmark_commons import clustering_algorithms
from src.benchmark.node_clustering.tu_dataset_benchmark.tu_dataset_benchmark import tu_dataset_benchmark
from src.topological_embeddings.algorithms.common.rmrse_loss_function import rmrse_loss, count_rmrse_for
from src.topological_embeddings.algorithms.common.rmse_loss_function import rmse_loss, count_rmse_for
from src.topological_embeddings.algorithms.direct.linear_embeddings_optimization import get_direct_embeddings
from src.topological_embeddings.algorithms.neural.neural_embeddings_optimization import get_neural_embeddings

dataset_names = [
    "MUTAG",
    "Cuneiform",
    "SYNTHETIC",
    "ENZYMES",
    "IMDB-BINARY"
]
embedding_sizes = [2, 3, 5, 10, 15, 30, 50]
kappas = [None, 1.0, 1.5]
loss_functions = [[rmse_loss, "RMSE", count_rmse_for], [rmrse_loss, "RMRSE", count_rmrse_for]]
embedding_algorithms = [[get_neural_embeddings, "neural"], [get_direct_embeddings, "direct"]]

benchmark_dir = "/home/radekpriv/wyniki"

for kappa in kappas:
    for dataset_name in dataset_names:
        for [[loss_fun, loss_fun_name, evaluate_fun], [embedding_algorithm, embedding_algorithm_name]] in cartesian(
                loss_functions, embedding_algorithms
        ):
            tu_dataset_benchmark(
                dataset_name,
                embedding_algorithm=loss_fun,
                clustering_algorithms=clustering_algorithms,
                output_csv_filename=r"%s/kappa%s/%s_%s_transform_%s.csv" % (
                    benchmark_dir, kappa, loss_fun_name, embedding_algorithm_name, dataset_name
                ),
                embedding_sizes=embedding_sizes,
                error_function=evaluate_fun,
                loss_function=loss_fun_name,
                kappa=kappa
            )
