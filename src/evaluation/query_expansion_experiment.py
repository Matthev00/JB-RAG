import time
from pathlib import Path

import wandb
from src.config import EMBEDDING_MODEL
from src.evaluation.dataset import RAGDataset
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.utils import set_seeds
from src.retriever.faiss_search import FAISSRetriever


def main():
    """
    Script for running Query Expansion experiments.
    """
    set_seeds(42)

    dataset = RAGDataset(Path("data/escrcpy_val.json"))

    def evaluate_query_expansion(
        expand_query_type: str, retriever_params: dict, trial_number
    ):
        """
        Evaluates the system with a specific Query Expansion type.

        Args:
            expand_query_type (str): Type of Query Expansion (None, candidate_terms, wordnet).
            retriever_params (dict): Parameters for the retriever (radius or top_k).
            trail_number (int): idx of trail

        Returns:
            dict: Evaluation results including quality metrics and latency.
        """
        wandb.init(
            project="JB-RAG-query-expansion",
            name=f"trial-{trial_number}",
            reinit=True,
        )
        retriever = FAISSRetriever(EMBEDDING_MODEL)
        retriever.load_index("escrcpy")

        start_time = time.time()
        results = RAGEvaluator.evaluate(
            retriever, dataset, expand_query_type=expand_query_type, **retriever_params
        )
        latency = time.time() - start_time

        wandb.log(
            {
                "expand_query_type": expand_query_type,
                "latency": latency,
                "radius": retriever_params.get("radius"),
                "Precision@10": results["Precision@10"],
                "Recall@10": results["Recall@10"],
                "F1@10": results["F1@10"],
            }
        )

        return results

    query_expansion_types = [None, "candidate_terms", "wordnet"]
    retriever_configs = [
        {"radius": 0.3, "top_k": None},
        {"radius": 0.05, "top_k": None},
    ]
    trial_number = 1
    for expand_query_type in query_expansion_types:
        for retriever_params in retriever_configs:
            evaluate_query_expansion(expand_query_type, retriever_params, trial_number)
            trial_number += 1

    wandb.finish()


if __name__ == "__main__":
    main()
