from pathlib import Path

from src.config import EMBEDDING_MODEL
from src.evaluation.dataset import RAGDataset
from src.retriever.faiss_search import FAISSRetriever


class RAGEvaluator:
    @staticmethod
    def evaluate(retriever: FAISSRetriever, dataset: RAGDataset, **retriever_params):
        """
        Evaluates the retriever on the given dataset.

        Args:
            retriever (FAISSRetriever): Retriever object.
            dataset (RAGDataset): Dataset object.
            retriever_params: Parameters for the retriever.

        Returns:
            dict: Evaluation results.
        """
        precision_scores = []
        recall_scores = []
        f1_scores = []
        mrr_scores = []

        for question, expected_files in dataset:
            results = retriever.search(question, **retriever_params)
            retrieved_files = set([res["relative_path"] for res in results])

            # Precision@10
            precision = (
                len(retrieved_files & expected_files) / len(retrieved_files)
                if retrieved_files
                else 0
            )
            precision_scores.append(precision)

            # Recall@10
            recall = (
                len(retrieved_files & expected_files) / len(expected_files)
                if expected_files
                else 0
            )
            recall_scores.append(recall)

            # F1@10
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            f1_scores.append(f1)

            # MRR
            rank = next(
                (
                    i + 1
                    for i, res in enumerate(results)
                    if res["relative_path"] in expected_files
                ),
                0,
            )
            mrr_scores.append(1 / rank if rank > 0 else 0)
        return {
            "Precision@10": sum(precision_scores) / len(precision_scores),
            "Recall@10": sum(recall_scores) / len(recall_scores),
            "F1@10": sum(f1_scores) / len(f1_scores),
            "MRR": sum(mrr_scores) / len(mrr_scores),
        }
