import numpy as np
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.preprocessing.code_parser import CodeParser


class EmbeddingGenerator:
    """ Creates embeddings for code chunks using SentenceTransformer """

    def __init__(self, model_name:str = "all-MiniLM-L6-v2") -> None:
        """
        Initializes the SentenceTransformer model.

        Args:
            model_name (str): Name of the SentenceTransformer model."""
        self.model = SentenceTransformer(model_name)

    def create_embeddings(self, code_chunks: list[dict]) -> list[dict]:
        """
        Creates embeddings for code chunks

        Args:
            code_chunks (list): List of code chunks with metadata.
        
        Returns:
            list: List of code chunks with added embeddings.
        """
        texts = [chunk["code"].strip() for chunk in code_chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        for i, chunk in enumerate(code_chunks):
            chunk["embedding"] = embeddings[i]

        return code_chunks

    def save_embeddings(self, code_chunks: list[dict], save_path: Path) -> None:
        """
        Saves embeddings and metadata to disk.
        Embeddings are saved as a NumPy array, metadata is saved as JSON.
        
        Args:
            code_chunks (list): List of code chunks with metadata and embeddings.
            save_path (Path): Path to save the embeddings and metadata.
        """

        save_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(save_path.with_suffix(".npy"), np.array([c["embedding"] for c in code_chunks]))

        metadata = [{k: v for k, v in c.items() if k != "embedding"} for c in code_chunks]
        with save_path.with_suffix(".json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_embeddings(self, save_path: Path) -> list[dict]:
        """
        Loads embeddings and metadata from disk.
        
        Args:
            save_path (Path): Path to the embeddings and metadata.

        Returns:
            list: List of code chunks with metadata and embeddings.
        """
        metadata_path = (save_path / "metadata").with_suffix(".json")
        embeddings_path = (save_path / "embeddings").with_suffix(".npy")

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        embeddings = np.load(embeddings_path)

        for i, chunk in enumerate(metadata):
            chunk["embedding"] = embeddings[i]

        return metadata


if __name__ == "__main__":
    save_path = Path("data/embeddings") / "escrcpy"

    repo_path = Path("data/repos/escrcpy")
    code_parser = CodeParser(repo_path)
    code_chunks = code_parser.parse(30)

    embedding_generator = EmbeddingGenerator()
    code_chunks = embedding_generator.create_embeddings(code_chunks)
    embedding_generator.save_embeddings(code_chunks, save_path)

    loaded_chunks = embedding_generator.load_embeddings(save_path)
    print(loaded_chunks[:15])
