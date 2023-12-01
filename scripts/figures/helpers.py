import os
import pickle
import pandas as pd
from typing import Dict, List, Union, Callable
from functools import partial


from scripts.utils import main_experiment_results_dir, overriding_experiment_results_dir, attention_experiment_results_dir, similarity_experiment_results_dir
from core.config import FIGURES_DIR

MODEL_DISPLAY_NAME_MAPPING = {
    "llama_7B": "LLaMA 7B",
    "llama_13B": "LLaMA 13B",
    "llama_30B": "LLaMA 30B",
    "gpt-j_6B": "GPT-J 6B",
    "pythia_2.8B": "Pythia 2.8B",
    "pythia_6.9B": "Pythia 6.9B",
    "pythia_12B": "Pythia 12B",
}


def load_results(get_dir: Callable, experiment_id: str = "camera_ready"):
    results = {}
    results_dir = get_dir(experiment_id)

    for results_file in os.listdir(results_dir):
        model_name = results_file[:-4]
        file_path = os.path.join(results_dir, results_file)
        with open(file_path, "rb") as f:
            results[model_name] = pickle.load(f)

    return results


load_main_results = partial(load_results, main_experiment_results_dir)
load_overriding_results = partial(load_results, overriding_experiment_results_dir)
load_attention_results = partial(load_results, attention_experiment_results_dir)
load_similarity_results = partial(load_results, similarity_experiment_results_dir)


def get_only_last_token_attentions(results: Dict[str, Dict[str, Dict[str, Union[List[List[str]], List[List[List[float]]]]]]]):
    filtered_results = {}
    
    for model_name, model_results in results.items():
        filtered_results[model_name] = {}
        for task_name, task_results in model_results.items():
            filtered_results[model_name][task_name] = {}

            filtered_results[model_name][task_name]["tokenized_text"] = [[token.replace('Ġ', '_').replace('Ċ', '<newline>') for token in dataset_text] for dataset_text in task_results["tokenized_text"]]
            filtered_results[model_name][task_name]["attention"] = task_results["attention"]
            filtered_results[model_name][task_name]["last_tok_attn"] = [
                [
                    dataset_attention[-1] for dataset_idx, dataset_attention in enumerate(layer_attention)
                ]
                for layer_attention in task_results["attention"]
            ]

    return filtered_results


def filter_attention_results(results: Dict[str, Dict[str, Dict[str, Union[List[List[str]], List[List[List[float]]]]]]]):
    filtered_results = {}
    
    for model_name, model_results in results.items():
        filtered_results[model_name] = {}
        for task_name, task_results in model_results.items():
            filtered_results[model_name][task_name] = {}

            filtered_query_indices = [[i for i, token in enumerate(dataset_text) if token == '->'] for dataset_text in task_results["tokenized_text"]]

            filtered_results[model_name][task_name]["tokenized_text"] = [[token.replace('Ġ', '_').replace('Ċ', '<newline>') for token in dataset_text] for dataset_text in task_results["tokenized_text"]]
            filtered_results[model_name][task_name]["attention"] = [
                [
                    [
                        [
                            query_attention for query_attention in dataset_attention[token_query_indices]
                        ]
                        for token_query_indices in filtered_query_indices[dataset_idx]
                    ]
                    for dataset_idx, dataset_attention in enumerate(layer_attention)
                ]
                for layer_idx, layer_attention in enumerate(task_results["attention"])
            ]
            filtered_results[model_name][task_name]["queries"] = [
                [task_results["tokenized_text"][dataset_idx][query_index] for query_index in query_indices] for dataset_idx, query_indices in enumerate(filtered_query_indices)
            ]

    return filtered_results


def extract_accuracies(results):
    accuracies = {}
    for model_name, model_results in results.items():
        accuracies[model_name] = {}
        for task_name, task_results in model_results.items():
            accuracies[model_name][task_name] = {
                "bl": task_results["baseline_accuracy"],
                "icl": task_results["icl_accuracy"],
                "tv": task_results["tv_accuracy"],
            }

    return accuracies


def create_accuracies_df(results):
    accuracies = extract_accuracies(results)

    data = []
    for model_name, model_acc in accuracies.items():
        for task_full_name, task_acc in model_acc.items():
            task_type = task_full_name.split("_")[0]
            task_name = "_".join(task_full_name.split("_")[1:])

            data.append([model_name, task_type, task_name, "Baseline", task_acc["bl"]])
            data.append([model_name, task_type, task_name, "Hypothesis", task_acc["tv"]])
            data.append([model_name, task_type, task_name, "Regular", task_acc["icl"]])

    df = pd.DataFrame(data, columns=["model", "task_type", "task_name", "method", "accuracy"])

    df["model"] = df["model"].map(MODEL_DISPLAY_NAME_MAPPING)

    # order the tasks by alphabetical order, using the task_full_name
    task_order = sorted(zip(df["task_type"].unique(), df["task_name"].unique()), key=lambda x: x[0])
    task_order = [x[1] for x in task_order]

    return df


def create_grouped_accuracies_df(accuracies_df):
    grouped_accuracies_df = accuracies_df.pivot_table(
        index=["model", "task_type", "task_name"], columns="method", values="accuracy", aggfunc="first"
    ).reset_index()
    return grouped_accuracies_df
