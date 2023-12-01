# This must be first
from dotenv import load_dotenv

load_dotenv(".env")

import sys
import os
import pickle
import time
import torch as th
from typing import Optional, List, Tuple

from transformers import PreTrainedModel, PreTrainedTokenizer

from scripts.utils import SIMILARITY_RESULTS_DIR, similarity_experiment_results_dir

from core.data.task_helpers import get_all_tasks, get_task_by_name
from core.models.llm_loading import load_model_and_tokenizer
from core.analysis.evaluation import calculate_accuracy_on_datasets
from core.task_vectors import run_intermediate_icl
from core.utils.misc import limit_gpus, seed_everything
from core.experiments_config import MODELS_TO_EVALUATE, TASKS_TO_EVALUATE


def get_results_file_path(model_type: str, model_variant: str, experiment_id: str = "") -> str:
    return os.path.join(similarity_experiment_results_dir(experiment_id), f"{model_type}_{model_variant}.pkl")


def get_task_vectors_and_accuracies(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, task_name: str, num_examples: int, num_dev_datasets: int = 50, num_test_datasets: int = 50) -> Tuple[List[List[List[List[float]]]], List[List[str]]]:
    seed_everything(42)

    task = get_task_by_name(tokenizer=tokenizer, task_name=task_name)
    
    test_datasets = task.create_subset_datasets(num_datasets=num_test_datasets, max_examples=num_examples)
    dev_datasets = task.create_datasets(num_datasets=num_dev_datasets, num_examples=num_examples)

    task = get_task_by_name(tokenizer=tokenizer, task_name=task_name)

    task_vectors, predictions = run_intermediate_icl(model, tokenizer, task, test_datasets, dev_datasets, num_examples)  # (num_examples, B, D), (num_examples, B)

    accuracies_by_length = [calculate_accuracy_on_datasets(task, prediction, test_datasets[i]) for i, prediction in enumerate(predictions)]  # (num_examples)

    mean_task_vectors = task_vectors.mean(dim=1)  # (num_examples, D)

    return mean_task_vectors, accuracies_by_length


def run_similarity_experiment(
    model_type: str,
    model_variant: str,
    experiment_id: str = "",
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> None:
    print("Pulling model task vector similarities and accuracies:", model_type, model_variant)

    results_file = get_results_file_path(model_type, model_variant, experiment_id=experiment_id)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    limit_gpus(range(0, 8))

    print("Loading model and tokenizer...")
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(model_type, model_variant)
    print("Loaded model and tokenizer.")

    tasks = get_all_tasks(tokenizer=tokenizer)

    num_examples = 5
    num_dev_datasets, num_test_datasets = 50, 50

    for i, task_name in enumerate(TASKS_TO_EVALUATE):
        if task_name in results:
            print(f"Skipping task {i+1}/{len(tasks)}: {task_name}")
            continue

        results[task_name] = {}

        print("\n" + "=" * 50)
        print(f"Getting similarities for task {i+1}/{len(tasks)}: {task_name}")

        task_vectors, accuracies = get_task_vectors_and_accuracies(model, tokenizer, task_name, num_examples, num_dev_datasets, num_test_datasets)

        results[task_name] = {
            "task_vectors": task_vectors,  # (num_examples, D)
            "accuracies": accuracies,  # (num_examples)
        }

        with open(results_file, "wb") as f:
            pickle.dump(results, f)


def get_new_experiment_id() -> str:
    return str(
        max([int(results_dir) for results_dir in os.listdir(SIMILARITY_RESULTS_DIR) if results_dir.isdigit()] + [0]) + 1
    )


def main():
    if len(sys.argv) == 1:
        # Run all models
        # Calculate the experiment_id as the max experiment_id + 1
        experiment_id = get_new_experiment_id()
        for model_type, model_variant in MODELS_TO_EVALUATE:
            run_similarity_experiment(model_type, model_variant, experiment_id=experiment_id)
    else:
        if len(sys.argv) == 2:
            model_num = int(sys.argv[1])
            model_type, model_variant = MODELS_TO_EVALUATE[model_num]
        elif len(sys.argv) == 3:
            model_type, model_variant = sys.argv[1:]

        run_similarity_experiment(model_type, model_variant)


if __name__ == "__main__":
    main()
