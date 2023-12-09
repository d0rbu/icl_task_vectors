import os
import torch as th
from scripts.figures.helpers import load_early_exit_results, MODEL_DISPLAY_NAME_MAPPING
from scripts.utils import EARLY_EXIT_FIGURES_DIR
from core.config import FIGURES_DIR
import matplotlib.pyplot as plt
from typing import Callable


def create_early_exit_figures(experiment_id: str = "camera_ready"):
    results = load_early_exit_results(experiment_id)

    for model_name, model_results in results.items():
        average_accuracies = th.zeros(len(list(model_results.values())[0]["accuracies"]))
        early_exit_dir = os.path.join(EARLY_EXIT_FIGURES_DIR, model_name)
        os.makedirs(early_exit_dir, exist_ok=True)

        for task_name, task_results in model_results.items():
            accuracies = th.tensor(task_results["accuracies"])

            average_accuracies += accuracies

            plt.plot(accuracies)
            plt.xlabel("Model Depth")
            plt.ylabel("Accuracy")

            plt.savefig(os.path.join(early_exit_dir, f"{task_name}.png"))
            plt.clf()

        average_accuracies /= len(model_results)

        plt.plot(average_accuracies)
        plt.title(f"{MODEL_DISPLAY_NAME_MAPPING[model_name]} Early Exit Accuracies By Layer")
        plt.xlabel("Model Depth")
        plt.ylabel("Accuracy")

        plt.savefig(os.path.join(EARLY_EXIT_FIGURES_DIR, f"{model_name}.png"))
        plt.clf()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    experiment_name = "camera_ready"

    create_early_exit_figures(experiment_name)


if __name__ == "__main__":
    main()
