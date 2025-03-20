from src.config import EMBEDDING_MODEL
from src.retriever.faiss_search import FAISSRetriever


def main():
    retriever = FAISSRetriever(embedding_model=EMBEDDING_MODEL)
    retriever.load_index("escrcpy")
    results = retriever.search(
        "How does the application determine and validate the paths for external binaries (like scrcpy, adb, and gnirehtet) at startup, and which module or file manages this logic?",
        radius=0.3,
    )
    for result in results:
        print(result["relative_path"])


if __name__ == "__main__":
    main()
