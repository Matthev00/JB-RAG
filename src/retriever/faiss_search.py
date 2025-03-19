import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, EMBEDDINGS_DIR, FAISS_INDEX_DIR


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

    def search(self, query: str, radius: float = 0.8) -> list[dict]:
        """
        Searches the FAISS index for all code chunks within a given similarity radius.

        Args:
            query (str): Query string.
            radius (float): Radius for similarity search (cosine similarity threshold).

        Returns:
            list: List of similar code chunks with metadata.
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        lims, distances, indices = self.index.range_search(query_embedding, radius)

        files = set()
        results = []
        for i in range(len(lims) - 1):
            for idx, dist in zip(
                indices[lims[i]: lims[i + 1]], distances[lims[i]: lims[i + 1]]
            ):
                if (
                    self.metadata[idx]["file_type"] == "code"
                    and self.metadata[idx]["relative_path"] not in files
                ):
                    results.append(self.metadata[idx])
                    files.add(self.metadata[idx]["relative_path"])

        return results

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
