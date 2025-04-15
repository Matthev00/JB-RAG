import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import EMBEDDINGS_DIR, FAISS_INDEX_DIR
from src.retriever.query_expander import QueryExpander


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

    def expand_query(
        self,
        query: str,
        expand_query_type: str,
        query_top_k: int,
    ) -> np.ndarray:
        """
        Expands the query using the specified method.

        Args:
            query (str): Original query.
            expand_query_type (str): Type of query expansion technique.
            query_top_k (int): Number of similar terms to add.

        Returns:
            np.ndarray: Embedding of the expanded query.
        """
        if expand_query_type == "llm_generated":
            query_embedding = QueryExpander.query_with_LLM(
                query=query, model=self.model
            )
        else:
            if expand_query_type == "wordnet":
                query = QueryExpander.expand_query_with_wordnet(query=query)
            elif expand_query_type == "candidate_terms":
                query = QueryExpander.expand_query_with_embeddings(
                    query=query, model=self.model, top_k=query_top_k
                )

            query_embedding = self.model.encode([query], convert_to_numpy=True)
        return query_embedding

    def search(
        self,
        query: str,
        radius: float = None,
        top_k: int = None,
        expand_query_type: str = None,
        rerank: bool = False,
        similarity_threshold: float = 0.5,
        query_top_k: int = 7,
    ) -> list[dict]:
        """
        Searches the FAISS index for code chunks based on either a similarity radius or top_k results.

        Args:
            query (str): Query string.
            radius (float, optional): Radius for similarity search .
            top_k (int, optional): Number of top results to retrieve.
            expand_query_type (str, optional): Type of expand query technique(None, wordnet, candidate_terms, llm_generated)
            rerank (bool): Use reranker or not
            similarity_threshold (float): similarity threshold for reranker
            query_top_k (int): top k for query expansion
            language (str): Language of the codebase (default is "JavaScript").

        Returns:
            list: List of similar code chunks with metadata.

        Raises:
            ValueError: If neither radius nor top_k is specified.
        """
        if radius is None and top_k is None:
            raise ValueError("Either 'radius' or 'top_k' must be specified.")

        query_embedding = self.expand_query(
            query=query,
            expand_query_type=expand_query_type,
            query_top_k=query_top_k,
        )

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

        if rerank:
            self.rerank(query_embedding, results, similarity_threshold)
        else:
            results.sort(key=lambda x: x[1])

        return [metadata for metadata, _ in results]

    def rerank(
        self,
        query_embedding: np.ndarray,
        results: list[dict],
        similarity_threshold: float,
    ) -> list[dict]:
        """
        Reranks the search results based on the cosine similarity between the query embedding and the code chunks.

        Args:
            query_embedding (np.ndarray): Precomputed embedding of the query.
            results (list): List of search results with metadata.
            similarity_threshold (float): Minimum cosine similarity score to include a result.

        Returns:
            list: List of reranked search results with metadata.
        """
        code_chunks = [res[0]["code"] for res in results]
        code_embeddings = self.model.encode(code_chunks, convert_to_numpy=True)

        scores = cosine_similarity(query_embedding, code_embeddings)[0]

        filtered_results = [
            (res, score)
            for res, score in zip(results, scores)
            if score >= similarity_threshold
        ]
        filtered_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)

        return filtered_results

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
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
