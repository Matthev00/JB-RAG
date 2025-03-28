from src.config import EMBEDDING_MODEL
from src.retriever.faiss_search import FAISSRetriever
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.dataset import RAGDataset
from pathlib import Path
from pprint import pprint


def main():
    retriever = FAISSRetriever(embedding_model=EMBEDDING_MODEL)
    retriever.load_index("escrcpy")
    dataset = RAGDataset(Path("data/escrcpy_val.json"))
    retriever_params = {"radius": 0.3, "top_k": None}
    results = RAGEvaluator.evaluate(retriever, dataset, **retriever_params)
    pprint(results)



if __name__ == "__main__":
    main()
