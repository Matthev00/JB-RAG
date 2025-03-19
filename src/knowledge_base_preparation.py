from pathlib import Path

from src.config import EMBEDDINGS_DIR, MAX_CHUNK_SIZE, REPO_URL
from src.preprocessing.code_parser import CodeParser
from src.preprocessing.embedding_generator import EmbeddingGenerator
from src.preprocessing.repo_loader import RepoLoader
from src.retriever.faiss_search import FAISSRetriever


def main():
    """
    Script for preparing the knowledge base.
    Includes:
        - cloning the repository
        - parsing the code
        - generating embeddings
        - building the FAISS index.

    Configurations can be found in src/config.py.
    """
    repo_loader = RepoLoader(REPO_URL)
    repo = repo_loader.clone()

    code_parser = CodeParser(Path(repo.working_dir))
    code_chunks = code_parser.parse(MAX_CHUNK_SIZE)

    project_name = repo.working_dir.split("/")[-1]

    embedding_generator = EmbeddingGenerator()
    code_chunks = embedding_generator.create_embeddings(code_chunks)
    embedding_generator.save_embeddings(
        code_chunks, save_path=Path(EMBEDDINGS_DIR) / project_name
    )

    retriever = FAISSRetriever()
    retriever.build_index(project_name)


if __name__ == "__main__":
    main()
