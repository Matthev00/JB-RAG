from pathlib import Path

from git import Repo

from src.config import REPO_DIR


class RepoLoader:
    def __init__(self, repo_url: str):
        """
        Class for downloding repo from external source.

        Args:
            repo_url (str): URL of the repository.
        """
        self.repo_url = repo_url
        self.repo_dir = Path(REPO_DIR) / Path(
            repo_url.split("/")[-1].replace(".git", "")
        )

    def clone(self) -> Repo:
        """
        Clones the repository from the URL to the local directory.
        Path is specified in config.py file as REPO_DIR.

        Returns:
            Repo: GitPython Repo object.
        """
        if self.repo_dir.exists():
            return Repo(self.repo_dir)
        else:
            return Repo.clone_from(self.repo_url, self.repo_dir)
