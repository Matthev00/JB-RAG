from src.config import EMBEDDING_MODEL
from src.retriever.faiss_search import FAISSRetriever


def selcet_type():
    print("Select the type of search you want to perform:")
    print("1. Radius search")
    print("2. Top K search")
    print("3. Exit")
    type = int(input())
    return type


def main():
    retriever = FAISSRetriever(embedding_model=EMBEDDING_MODEL)
    retriever.load_index("escrcpy")
    type = selcet_type()
    if type == 3:
        return
    while True:
        try:
            query = input("Enter your query: ")
            if query == "exit":
                break
            if type == 1:
                results = retriever.search(
                    query, radius=0.3, expand_query_type="wordnet"
                )
            elif type == 2:
                results = retriever.search(query, top_k=4, rerank=False)
            for result in results:
                print(result["relative_path"])
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
