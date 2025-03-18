from pathlib import Path

from git import Repo

from src.config import REPO_DIR


class RepoLoader:
    def __init__(self, repo_url: str):
        self.repo_url = repo_url
        self.repo_dir = Path(REPO_DIR) / Path(
            repo_url.split("/")[-1].replace(".git", "")
        )

    def clone(self) -> Repo:
        if self.repo_dir.exists():
            return Repo(self.repo_dir)
        else:
            return Repo.clone_from(self.repo_url, self.repo_dir)


def main():
    repo_url = "https://github.com/viarotel-org/escrcpy.git"
    repo_loader = RepoLoader(repo_url)
    repo = repo_loader.clone()
    print(repo)


if __name__ == "__main__":
    main()
