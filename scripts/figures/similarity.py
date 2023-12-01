import os
import pandas as pd
import torch as th
from scripts.figures.helpers import load_similarity_results
from scripts.utils import similarity_figures_dir
from core.config import FIGURES_DIR
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable


def create_similarity_figures(experiment_id: str = "camera_ready", similarity_metric: Callable, normalize: True):
    results = load_similarity_results(experiment_id)

    for model_name, model_results in results.items():
        for task_name, task_results in model_results.items():
            similarity_dir = similarity_figures_dir(model_name, task_name)
            os.makedirs(similarity_dir, exist_ok=True)
            task_vectors = task_results["task_vectors"]
            accuracies = task_results["accuracies"]
            num_max_examples = len(task_vectors)

            for num_examples in range(num_max_examples)
                current_task_vectors = task_vectors[num_examples]
                current_accuracy = accuracies[num_examples]

                plt.figure(figsize=(16, 8))
                sns.heatmap(dataset_attention_across_layers, xticklabels=dataset_tokens)
                plt.savefig(os.path.join(attention_dir, f"dataset={dataset_idx}.png"))
                plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    experiment_name = "camera_ready"

    create_similarity_figures(experiment_name)


if __name__ == "__main__":
    main()
