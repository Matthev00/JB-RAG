from tqdm import tqdm
import wandb

from src.evaluation.dataset import RAGDataset
from src.retriever.faiss_search import FAISSRetriever


class RAGEvaluator:
    @staticmethod
    def evaluate(
        retriever: FAISSRetriever,
        dataset: RAGDataset,
        retriever_params: dict,
        log_to_wandb: bool = False,
        wandb_run_name: str = None,
    ) -> dict:
        """
        Evaluates the retriever on the given dataset.

        Args:
            retriever (FAISSRetriever): Retriever object.
            dataset (RAGDataset): Dataset object.
            retriever_params (dict): Parameters for the retriever.
            log_to_wandb (bool): Whether to log results to Weights & Biases.
            wandb_run_name (str): Optional run name for wandb.

        Returns:
            dict: Evaluation results.
        """
        precision_scores = []
        recall_scores = []
        f1_scores = []
        mrr_scores = []

        for question, expected_files in tqdm(dataset):
            results = retriever.search(question, **retriever_params)
            top_10_results = results[:10]
            retrieved_files = set(res["relative_path"] for res in top_10_results)

            # Precision@10
            precision = (
                len(retrieved_files & expected_files) / len(retrieved_files)
                if retrieved_files else 0
            )
            precision_scores.append(precision)

            # Recall@10
            recall = (
                len(retrieved_files & expected_files) / len(expected_files)
                if expected_files else 0
            )
            recall_scores.append(recall)

            # F1@10
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0 else 0
            )
            f1_scores.append(f1)

            # MRR
            rank = next(
                (
                    i + 1
                    for i, res in enumerate(top_10_results)
                    if res["relative_path"] in expected_files
                ),
                0,
            )
            mrr_scores.append(1 / rank if rank > 0 else 0)

        # Final metrics
        metrics = {
            "Precision@10": sum(precision_scores) / len(precision_scores),
            "Recall@10": sum(recall_scores) / len(recall_scores),
            "F1@10": sum(f1_scores) / len(f1_scores),
            "MRR": sum(mrr_scores) / len(mrr_scores),
        }

        if log_to_wandb:
            wandb.init(
                project="code-rag",
                name=wandb_run_name or "retrieval_eval",
                config=retriever_params,
            )
            wandb.log(metrics)
            wandb.finish()

        return metrics

