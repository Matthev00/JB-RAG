import random
from pathlib import Path

import numpy as np
import optuna
import torch

import wandb
from src.config import EMBEDDING_MODEL
from src.evaluation.dataset import RAGDataset
from src.evaluation.evaluator import RAGEvaluator
from src.knowledge_base_preparation import prepare_knowledge_base
from src.retriever.faiss_search import FAISSRetriever


def set_seeds(seed: int):
    """
    Set seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    optuna.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """
    Script for running the optimization experiment with Optuna.
    """
    set_seeds(42)
    wandb.init(project="JB-RAG-optimization", name="optuna-experiment")

    dataset = RAGDataset(Path("data/escrcpy.json"))

    def objective(trial):
        """
        Objective function for Optuna to optimize parameters.
        """
        max_chunk_size = trial.suggest_int("max_chunk_size", 10, 100)
        use_radius = trial.suggest_categorical("use_radius", [True, False])
        if use_radius:
            radius = trial.suggest_float("radius", 0.05, 1.0)
            top_k = None
        else:
            radius = None
            top_k = trial.suggest_int("top_k", 1, 50)

        prepare_knowledge_base(max_chunk_size, "data/repos/escrcpy")
        retriever = FAISSRetriever(EMBEDDING_MODEL)
        retriever.build_index("escrcpy")
        retriever.load_index("escrcpy")

        retriever_params = {"radius": radius, "top_k": top_k}
        results = RAGEvaluator.evaluate(retriever, dataset, **retriever_params)

        wandb.log(
            {
                "max_chunk_size": max_chunk_size,
                "radius": radius,
                "top_k": top_k,
                "Precision@10": results["Precision@10"],
                "Recall@10": results["Recall@10"],
                "F1@10": results["F1@10"],
                "MRR": results["MRR"],
            }
        )

        return results["Recall@10"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2)

    wandb.log({"best_params": study.best_params, "best_Recall@10": study.best_value})

    wandb.finish()

    print("Best parameters:", study.best_params)
    print("Best Recall@10:", study.best_value)


if __name__ == "__main__":
    main()
