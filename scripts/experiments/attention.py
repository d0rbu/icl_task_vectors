# This must be first
from dotenv import load_dotenv

load_dotenv(".env")

import sys
import os
import pickle
import time
from typing import Optional, List, Tuple

from transformers import PreTrainedModel, PreTrainedTokenizer

from scripts.utils import ATTENTION_RESULTS_DIR, attention_experiment_results_dir

from core.data.task_helpers import get_all_tasks, get_task_by_name
from core.models.llm_loading import load_model_and_tokenizer
from core.models.utils.inference import hidden_to_logits
from core.analysis.utils import logits_top_tokens
from core.analysis.evaluation import calculate_accuracy_on_datasets
from core.task_vectors import run_icl, run_task_vector
from core.utils.misc import limit_gpus, seed_everything
from core.experiments_config import MODELS_TO_EVALUATE, TASKS_TO_EVALUATE


def get_results_file_path(model_type: str, model_variant: str, experiment_id: str = "") -> str:
    return os.path.join(attention_experiment_results_dir(experiment_id), f"{model_type}_{model_variant}.pkl")


def get_task_attention(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, task_name: str, num_examples: int, num_datasets: int = 1) -> Tuple[List[List[List[List[float]]]], List[List[str]]]:
    seed_everything(42)

    task = get_task_by_name(tokenizer=tokenizer, task_name=task_name)

    datasets = task.create_datasets(num_datasets=num_datasets, num_examples=num_examples)
    icl_token_sequences, icl_attention = run_icl(model, tokenizer, task, datasets, output_attentions=True)

    return icl_attention, icl_token_sequences


def run_attention_experiment(
    model_type: str,
    model_variant: str,
    experiment_id: str = "",
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> None:
    print("Pulling model attention:", model_type, model_variant)

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

    num_examples = 8
    num_datasets = 5

    for i, task_name in enumerate(TASKS_TO_EVALUATE):
        if task_name in results:
            print(f"Skipping task {i+1}/{len(tasks)}: {task_name}")
            continue

        results[task_name] = {}

        print("\n" + "=" * 50)
        print(f"Getting attentions for task {i+1}/{len(tasks)}: {task_name}")
        print(f"{num_examples} examples in this task and {num_datasets} datasets used")

        attention, tokenized_text = get_task_attention(model, tokenizer, task_name, num_examples, num_datasets)

        results[task_name] = {
            "attention": attention,  # (L, B, T, T)
            "tokenized_text": tokenized_text,  # (B, T)
        }

        with open(results_file, "wb") as f:
            pickle.dump(results, f)


def get_new_experiment_id() -> str:
    return str(
        max([int(results_dir) for results_dir in os.listdir(ATTENTION_RESULTS_DIR) if results_dir.isdigit()] + [0]) + 1
    )


def main():
    if len(sys.argv) == 1:
        # Run all models
        # Calculate the experiment_id as the max experiment_id + 1
        experiment_id = get_new_experiment_id()
        for model_type, model_variant in MODELS_TO_EVALUATE:
            run_attention_experiment(model_type, model_variant, experiment_id=experiment_id)
    else:
        if len(sys.argv) == 2:
            model_num = int(sys.argv[1])
            model_type, model_variant = MODELS_TO_EVALUATE[model_num]
        elif len(sys.argv) == 3:
            model_type, model_variant = sys.argv[1:]

        run_attention_experiment(model_type, model_variant)


if __name__ == "__main__":
    main()
