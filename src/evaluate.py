from pathlib import Path
from pprint import pprint

from src.config import EMBEDDING_MODEL, PROJECT_NAME
from src.evaluation.dataset import RAGDataset
from src.evaluation.evaluator import RAGEvaluator
from src.retriever.faiss_search import FAISSRetriever


def run_experiment(
    retriever: FAISSRetriever,
    dataset: RAGDataset,
    name: str,
    retriever_params: dict,
) -> dict:
    print(f"\n==> Running experiment: {name}")
    results = RAGEvaluator.evaluate(
        retriever, 
        dataset,
        retriever_params=retriever_params,
        log_to_wandb=True,
        wandb_run_name=name
    )
    pprint(results)
    return {
        "experiment": name,
        "params": retriever_params,
        "metrics": results,
    }

def main():
    reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    retriever = FAISSRetriever(
        embedding_model=EMBEDDING_MODEL,
        reranker_model=reranker_model_name,
        )
    retriever.load_index(PROJECT_NAME)
    dataset = RAGDataset(Path("data/escrcpy_val.json"))

    experiments = [
        {
            "name": "baseline_no_rerank-ms-macro",
            "params": {
                "radius": None,
                "top_k": 50,
                "expand_query_type": "candidate_terms",
                "rerank": True,
            },
        },
        {
            "name": "baseline_with_rerank-ms-macro",
            "params": {
                "radius": None,
                "top_k": 20,
                "expand_query_type": "candidate_terms",
                "rerank": True,
            },
        },
        {
            "name": "rerank_top50-ms-macro",
            "params": {
                "radius": None,
                "top_k": 50,
                "expand_query_type": "candidate_terms",
                "rerank": True,
            },
        },
        {
            "name": "rerank_wordnet-ms-macro",
            "params": {
                "radius": None,
                "top_k": 20,
                "expand_query_type": "wordnet",
                "rerank": True,
            },
        },
        {
            "name": "no_rerank_llm_expand-ms-macro",
            "params": {
                "radius": None,
                "top_k": 10,
                "expand_query_type": "llm_generated",
                "rerank": False,
            },
        },
        {
            "name": "rerank_llm_expand-ms-macro",
            "params": {
                "radius": None,
                "top_k": 20,
                "expand_query_type": "llm_generated",
                "rerank": True,
            },
        },
    ]

    all_results = []

    for exp in experiments:
        result = run_experiment(retriever, dataset, exp["name"], exp["params"])
        all_results.append(result)

    print("\n==> All experiments completed.\nSummary:")
    for r in all_results:
        print(f"{r['experiment']}: {r['metrics']}")


if __name__ == "__main__":
    main()
