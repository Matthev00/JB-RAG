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
        "top_k": 11,
        "expand_query_type": "llm_generated",
    }
    results = RAGEvaluator.evaluate(retriever, dataset, **retriever_params)
    pprint(results)


if __name__ == "__main__":
    main()
