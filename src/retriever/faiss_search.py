import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json

from src.config import FAISS_INDEX_DIR, EMBEDDINGS_DIR, EMBEDDING_MODEL


class FAISSRetriever:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2") -> None:
        """
        Initializes the SentenceTransformer model and the FAISS index.

        Args:
            embedding_model (str): Name of the SentenceTransformer model.
        """
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.metadata = []

    def load_index(self, project_name: str) -> None:
        """
        Loads the FAISS index and metadata from disk.

        Args:
            project_name (str): Name of the project.
        """
        index_path = (Path(FAISS_INDEX_DIR) / project_name).with_suffix(".faiss")
        metadata_path = (Path(EMBEDDINGS_DIR) / project_name / "metadata").with_suffix(
            ".json"
        )

        self.index = faiss.read_index(str(index_path))

        with metadata_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Searches the FAISS index for the most similar code chunks.

        Args:
            query (str): Query string.
            k (int): Number of similar code chunks to retrieve.

        Returns:
            list: List of similar code chunks with metadata.
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        _, indices = self.index.search(query_embedding, k)

        return [self.metadata[i] for i in indices[0]]

    def build_index(self, project_name: str) -> None:
        """
        Builds the FAISS index for the project.

        Args:
            project_name (str): Name of the project.
        """
        embeddings_path = (
            Path(EMBEDDINGS_DIR) / project_name / "embeddings"
        ).with_suffix(".npy")
        metadata_path = (Path(EMBEDDINGS_DIR) / project_name / "metadata").with_suffix(
            ".json"
        )

        embeddings = np.load(embeddings_path)
        self.metadata = json.loads(metadata_path.read_text())

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        index_path = (Path(FAISS_INDEX_DIR) / project_name).with_suffix(".faiss")
        faiss.write_index(self.index, str(index_path))


if __name__ == "__main__":
    retriever = FAISSRetriever(EMBEDDING_MODEL)
    retriever.build_index("escrcpy")
    retriever.load_index("escrcpy")
    results = retriever.search("How does the SelectDisplay component handle the device options when retrieving display IDs?", 1)
    for result in results:
        print(result["path"])
        print()
        print("-" * 80)