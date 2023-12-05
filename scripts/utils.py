import os
from core.config import RESULTS_DIR, FIGURES_DIR

MAIN_RESULTS_DIR = os.path.join(RESULTS_DIR, "main")
OVERRIDING_RESULTS_DIR = os.path.join(RESULTS_DIR, "overriding")
ATTENTION_RESULTS_DIR = os.path.join(RESULTS_DIR, "attention")
ATTENTION_FIGURES_DIR = os.path.join(FIGURES_DIR, "attention")
SIMILARITY_RESULTS_DIR = os.path.join(RESULTS_DIR, "similarity")
SIMILARITY_FIGURES_DIR = os.path.join(FIGURES_DIR, "similarity")
EARLY_EXIT_RESULTS_DIR = os.path.join(RESULTS_DIR, "early_exit")
EARLY_EXIT_FIGURES_DIR = os.path.join(FIGURES_DIR, "early_exit")


def main_experiment_results_dir(experiment_id: str) -> str:
    return os.path.join(MAIN_RESULTS_DIR, experiment_id)


def overriding_experiment_results_dir(experiment_id: str) -> str:
    return os.path.join(OVERRIDING_RESULTS_DIR, experiment_id)


def attention_experiment_results_dir(experiment_id: str) -> str:
    return os.path.join(ATTENTION_RESULTS_DIR, experiment_id)


def attention_figures_dir(model_name: str, task_name: str) -> str:
    return os.path.join(ATTENTION_FIGURES_DIR, model_name, task_name)


def similarity_experiment_results_dir(experiment_id: str) -> str:
    return os.path.join(SIMILARITY_RESULTS_DIR, experiment_id)


def early_exit_experiment_results_dir(experiment_id: str) -> str:
    return os.path.join(EARLY_EXIT_RESULTS_DIR, experiment_id)
