import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDINGS_DIR, FAISS_INDEX_DIR


class FAISSRetriever:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2") -> None:
        """
        Initializes the SentenceTransformer model and the FAISS index.

        Args:
            embedding_model (str): Name of the SentenceTransformer model.
        """
        self.model = SentenceTransformer(embedding_model)
        self.index: faiss = None
        self.metadata: list[dict] = []

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

    def search(self, query: str, radius: float = None, top_k: int = None) -> list[dict]:
        """
        Searches the FAISS index for code chunks based on either a similarity radius or top_k results.

        Args:
            query (str): Query string.
            radius (float, optional): Radius for similarity search .
            top_k (int, optional): Number of top results to retrieve.

        Returns:
            list: List of similar code chunks with metadata.
        """
        if radius is None and top_k is None:
            raise ValueError("Either 'radius' or 'top_k' must be specified.")

        query_embedding = self.model.encode([query], convert_to_numpy=True)

        results = []
        files = set()

        if radius is not None:
            lims, distances, indices = self.index.range_search(query_embedding, radius)
            for i in range(len(lims) - 1):
                for idx, dist in zip(
                    indices[lims[i] : lims[i + 1]], distances[lims[i] : lims[i + 1]]
                ):
                    if (
                        self.metadata[idx]["file_type"] != "other"
                        and self.metadata[idx]["relative_path"] not in files
                    ):
                        results.append((self.metadata[idx], dist))
                        files.add(self.metadata[idx]["relative_path"])
        elif top_k is not None:
            distances, indices = self.index.search(query_embedding, top_k)
            for idx, dist in zip(indices[0], distances[0]):
                if (
                    self.metadata[idx]["file_type"] != "other"
                    and self.metadata[idx]["relative_path"] not in files
                ):
                    results.append((self.metadata[idx], dist))
                    files.add(self.metadata[idx]["relative_path"])

        results.sort(key=lambda x: x[1])

        return [metadata for metadata, _ in results]

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
