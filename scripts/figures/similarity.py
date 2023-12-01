import os
import torch as th
from scripts.figures.helpers import load_similarity_results
from scripts.utils import SIMILARITY_FIGURES_DIR
from core.config import FIGURES_DIR
import matplotlib.pyplot as plt
from typing import Callable


def cosine_similarity(x: th.Tensor, y: th.Tensor) -> float:
    if x.norm() == 0 or y.norm() == 0:
        return 0.0

    return x.dot(y) / (x.norm() * y.norm())


def dot_product_similarity(x: th.Tensor, y: th.Tensor) -> float:
    return x.dot(y)


def create_similarity_figures(experiment_id: str = "camera_ready", similarity_metric: Callable[[th.Tensor, th.Tensor], float] = cosine_similarity, normalize: bool = True):
    results = load_similarity_results(experiment_id)

    for model_name, model_results in results.items():
        average_similarities = th.zeros(len(list(model_results.values())[0]["accuracies"]))
        average_accuracies = average_similarities.clone()

        for task_name, task_results in model_results.items():
            similarity_dir = os.path.join(SIMILARITY_FIGURES_DIR, model_name)
            os.makedirs(similarity_dir, exist_ok=True)
            task_vectors = task_results["task_vectors"]
            accuracies = th.tensor(task_results["accuracies"])
            num_max_examples = task_vectors.shape[0]
            
            if normalize:
                task_vectors = task_vectors - task_vectors[0].unsqueeze(0)
            
            similarities_to_end = th.tensor([similarity_metric(task_vectors[i], task_vectors[-1]) for i in range(num_max_examples)])  # (num_examples)

            average_similarities += similarities_to_end
            average_accuracies += accuracies

            plt.plot(similarities_to_end, accuracies)
            plt.xlabel("Similarity to Task Vector")
            plt.ylabel("Accuracy")

            for i, (point_x, point_y) in enumerate(zip(similarities_to_end, accuracies)):
                plt.annotate(i, (point_x - 0.03, point_y + 0.008))

            plt.savefig(os.path.join(similarity_dir, f"{task_name}.png"))
            plt.clf()
        
        average_similarities /= len(model_results)
        average_accuracies /= len(model_results)

        plt.plot(average_similarities, average_accuracies)
        plt.xlabel("Similarity to Task Vector")
        plt.ylabel("Accuracy")

        for i, (point_x, point_y) in enumerate(zip(average_similarities, average_accuracies)):
            plt.annotate(i, (point_x - 0.03, point_y + 0.008))

        plt.savefig(os.path.join(SIMILARITY_FIGURES_DIR, f"{model_name}.png"))
        plt.clf()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    experiment_name = "camera_ready"

    create_similarity_figures(experiment_name)


if __name__ == "__main__":
    main()
