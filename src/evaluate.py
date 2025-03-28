from pathlib import Path
from pprint import pprint

from src.config import EMBEDDING_MODEL
from src.evaluation.dataset import RAGDataset
from src.evaluation.evaluator import RAGEvaluator
from src.retriever.faiss_search import FAISSRetriever


def main():
    retriever = FAISSRetriever(embedding_model=EMBEDDING_MODEL)
    retriever.load_index("escrcpy")
    dataset = RAGDataset(Path("data/escrcpy_val.json"))
    retriever_params = {"radius": 0.3, "top_k": None, "query_top_k": 5}
    results = RAGEvaluator.evaluate(
        retriever, dataset, expand_query_type="candidate_terms", **retriever_params
    )
    pprint(results)


if __name__ == "__main__":
    main()
