from pathlib import Path

from src.config import EMBEDDINGS_DIR, MAX_CHUNK_SIZE, REPO_URL
from src.preprocessing.code_parser import CodeParser
from src.preprocessing.embedding_generator import EmbeddingGenerator
from src.preprocessing.repo_loader import RepoLoader
from src.retriever.faiss_search import FAISSRetriever


def download_repo(repo_url: str) -> str:
    """
    Downloads the repository from the given URL.
    """
    repo_loader = RepoLoader(repo_url)
    repo = repo_loader.clone()
    return repo.working_dir


def prepare_knowledge_base(max_chunk_size: int, repo_path: str) -> None:
    """
    Script for preparing the knowledge base.
    Includes:
        - parsing the code
        - generating embeddings
        - building the FAISS index.

    Configurations can be found in src/config.py.

    Args:
        max_chunk_size (int): Maximum size of the code chunks
        repo_path (str): Path to the repository

    Returns:
        None
    """

    code_parser = CodeParser(Path(repo_path))
    code_chunks = code_parser.parse(max_chunk_size)

    project_name = repo_path.split("/")[-1]

    embedding_generator = EmbeddingGenerator()
    code_chunks = embedding_generator.create_embeddings(code_chunks)
    embedding_generator.save_embeddings(
        code_chunks, save_path=Path(EMBEDDINGS_DIR) / project_name
    )

    retriever = FAISSRetriever()
    retriever.build_index(project_name)


def main():
    repo_path = download_repo(REPO_URL)
    prepare_knowledge_base(MAX_CHUNK_SIZE, repo_path)


if __name__ == "__main__":
    main()
