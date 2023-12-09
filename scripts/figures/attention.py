import os
import torch as th
from scripts.figures.helpers import load_attention_results, filter_attention_results, get_only_last_token_attentions
from scripts.utils import attention_figures_dir
from core.config import FIGURES_DIR
import matplotlib.pyplot as plt
import seaborn as sns


def create_attention_figures(experiment_id: str = "camera_ready"):
    results = load_attention_results(experiment_id)
    # results = filter_attention_results(results)
    results = get_only_last_token_attentions(results)

    avg_arrow_to_query_layerwise_attns = {}

    for model_name, model_results in results.items():
        avg_arrow_to_query_layerwise_attn = th.zeros(len(list(model_results.values())[0]["attention"]), 2)
        num_examples = 0
        for task_name, task_results in model_results.items():
            attention_dir = attention_figures_dir(model_name, task_name)
            os.makedirs(attention_dir, exist_ok=True)
            num_datasets = len(task_results["attention"][0])

            for dataset_idx in range(num_datasets):
                start_idx = 0
                for i, token in enumerate(reversed(task_results["tokenized_text"][dataset_idx])):
                    if token == '<|endoftext|>' or token == '<s>':
                        start_idx = len(task_results["tokenized_text"][dataset_idx]) - i
                        break
                
                dataset_attention_across_layers = th.tensor([
                    layer_attention[dataset_idx][start_idx + 1:]
                    for layer_attention in task_results["last_tok_attn"]
                ])
                dataset_tokens = task_results["tokenized_text"][dataset_idx][start_idx + 1:]

                avg_arrow_to_query_layerwise_attn += dataset_attention_across_layers[:, -2:]

                # plt.figure(figsize=(16, 8))
                # sns.heatmap(dataset_attention_across_layers, xticklabels=dataset_tokens)
                # plt.savefig(os.path.join(attention_dir, f"dataset={dataset_idx}.png"))
                # plt.close()

                num_examples += 1
        
        avg_arrow_to_query_layerwise_attn /= num_examples
        avg_arrow_to_query_layerwise_attns[model_name] = avg_arrow_to_query_layerwise_attn
    
    fig, axs = plt.subplots(1, len(results), figsize=(8, 8))

    for i, (model_name, avg_arrow_to_query_layerwise_attn) in enumerate(avg_arrow_to_query_layerwise_attns.items()):
        sns.heatmap(avg_arrow_to_query_layerwise_attn, ax=axs[i])
        axs[i].set_xlabel(model_name)
    
    fig.suptitle("Average Attention to Query Token by Layer")
    plt.savefig(os.path.join(FIGURES_DIR, "attention_to_query.png"))


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    experiment_name = "camera_ready"

    create_attention_figures(experiment_name)


if __name__ == "__main__":
    main()
