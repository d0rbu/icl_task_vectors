import os
import pandas as pd
from scripts.figures.helpers import load_attention_results
from scripts.utils import attention_figures_dir
from core.config import FIGURES_DIR
import matplotlib.pyplot as plt
import seaborn as sns


def create_attention_figures(experiment_id: str = "camera_ready"):
    results = load_attention_results(experiment_id)

    for model_name, model_results in results.items():
        for task_name, task_results in model_results.items():
            attention_dir = attention_figures_dir(model_name, task_name)
            os.makedirs(attention_dir, exist_ok=True)

            for layer_idx, layer_attention in enumerate(task_results["attention"]):
                for dataset_idx, (dataset_tokens, dataset_attention) in enumerate(zip(task_results["tokenized_text"], layer_attention)):
                    plt.figure(figsize=(10, 10))
                    sns.heatmap(dataset_attention, xticklabels=dataset_tokens, yticklabels=dataset_tokens)
                    plt.savefig(os.path.join(attention_dir, f"dataset={dataset_idx}_layer={layer_idx}.png"))
                    plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    experiment_name = "camera_ready"

    create_attention_figures(experiment_name)


if __name__ == "__main__":
    main()
