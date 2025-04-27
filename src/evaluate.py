from pathlib import Path
from pprint import pprint

from src.config import EMBEDDING_MODEL, PROJECT_NAME
from src.evaluation.dataset import RAGDataset
from src.evaluation.evaluator import RAGEvaluator
from src.retriever.faiss_search import FAISSRetriever


def main():
    retriever = FAISSRetriever(embedding_model=EMBEDDING_MODEL)
    retriever.load_index(PROJECT_NAME)
    dataset = RAGDataset(Path("data/escrcpy_val.json"))
    retriever_params = {
        "radius": None,
        "top_k": 10,
        "expand_query_type": "candidate_terms",
        "rerank": False,
    }

    print("Evaluating WITHOUT reranker...")
    results_no_rerank = RAGEvaluator.evaluate(retriever, dataset, **retriever_params)
    pprint(results_no_rerank)

    retriever_params_rerank = {
        "radius": None,
        "top_k": 20,
        "expand_query_type": "candidate_terms",
        "rerank": True,
    }

    print("\nEvaluating WITH reranker...")
    results_rerank = RAGEvaluator.evaluate(retriever, dataset, **retriever_params_rerank)
    pprint(results_rerank)

if __name__ == "__main__":
    main()
