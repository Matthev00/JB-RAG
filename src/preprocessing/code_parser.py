import re
from pathlib import Path

from src.config import (
    CONFIG_EXTENSIONS,
    DEFAULT_IGNORED_EXTENSIONS,
    DOC_EXTENSIONS,
    LANGUAGE_PATTERNS,
)
from pprint import pprint


class CodeParser:
    def __init__(self, repo_path: Path) -> None:
        """
        Initialize the CodeParser object with the path to the repository.
        Sets the ignored patterns to gitignore patterns and default ignored extensions.

        Args:
            repo_path (Path): Path to the repository.
        """
        self._repo_path = repo_path
        self.ignored_patterns = (
            self._load_gitignore_patterns() | DEFAULT_IGNORED_EXTENSIONS
        )

    def _load_gitignore_patterns(self) -> set:
        """
        Load the patterns from the .gitignore file in the repository.

        Args:
            repo_path (Path): Path to the repository.
        Returns:
            set: Set of patterns to be ignored if present, else empty set.
        """
        gitignore_path = self._repo_path / ".gitignore"
        if not gitignore_path.exists():
            return set()

        with open(gitignore_path, "r") as f:
            lines = f.readlines()

        patterns = set()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("!"):
                # Negation
                line = line[1:]
                patterns.discard(line)
            else:
                patterns.add(line)

        return patterns

    def _is_file_relevant(self, file_path: Path) -> bool:
        """
        Check if the file is relevant for the analysis.

        Args:
            file (Path): Path to the file.
        Returns:
            bool: True if the file is relevant, else False.
        """
        if not file_path.is_file():
            return False
        if any(file_path.match(pattern) for pattern in self.ignored_patterns):
            return False

        try:
            with file_path.open("rb") as f:
                chunk = f.read(1024)
                if b"\0" in chunk:
                    return False
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                return bool(f.readline())
        except Exception:
            return False

    def get_relevant_files(self) -> list[Path]:
        """
        Get the code files in the repository.

        Returns:
            list[Path]: List of paths to the code files.
        """
        return [
            file for file in self._repo_path.rglob("*") if self._is_file_relevant(file)
        ]

    def _classify_file(self, file_path: Path) -> str:
        """
        Classify the file as code, config or documentation.

        Args:
            file_path (Path): Path to the file.

        Returns:
            str: Type of the file.(code, config, documentation)
        """
        ext = file_path.suffix
        if ext in CONFIG_EXTENSIONS:
            return "config"
        if ext in DOC_EXTENSIONS:
            return "documentation"
        return "code"

    def _detect_language(self, file_path: Path) -> str:
        """
        Detect the language of the code file.

        Args:
            file_path (Path): Path to the code file.

        Returns:
            str: Language of the code file.
        """
        return {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".vue": "Vue",
            ".cpp": "C++",
            ".java": "Java",
            ".html": "HTML",
            ".css": "CSS",
            ".sh": "Shell",
            ".sql": "SQL",
            ".r": "R",
            ".rb": "Ruby",
            ".php": "PHP",
            ".cs": "C#",
            ".go": "Go",
            ".md": "Markdown",
            ".yaml": "YAML",
            ".json": "JSON",
            ".xml": "XML",
        }.get(file_path.suffix, "Unknown")

    def split_file(
        self, file_path: Path, max_chunk_size: int, file_language: str
    ) -> list[tuple[str, int, int]]:
        """
        Splits file into chunks.

        Args:
            file_path (Path): Path to the code file.
            max_chunk_size (int): Maksimum chunk size.
            file_language (str): language of file.

        Returns:
            list[tuple[str, int, int]]: List of Tuples in each tuple code lines, start line and end line
        """
        block_pattern = LANGUAGE_PATTERNS.get(file_language)

        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        chunks = []
        current_chunk = []
        chunk_start_line = 0

        for i, line in enumerate(lines):
            current_chunk.append(line)

            if block_pattern and re.match(block_pattern, line):
                if len(current_chunk) >= max_chunk_size:
                    chunks.append(
                        (
                            "".join(current_chunk),
                            chunk_start_line,
                            i,
                        )
                    )
                    current_chunk = []
                    chunk_start_line = i

        if current_chunk:
            chunks.append(
                (
                    "".join(current_chunk),
                    chunk_start_line,
                    i,
                )
            )

        return chunks

    def parse(self, max_chunk_size: int) -> list[dict]:
        """
        Parse the code files in the repository.

        Args:
            max_chunk_size (int): Maximum size of the code chunk.
        Returns:
            list: List of code chunks.
        """
        all_chunks: list[dict] = []
        file_paths = self.get_relevant_files()
        i = 0
        for file in file_paths:
            file_type = self._classify_file(file)
            file_language = self._detect_language(file)
            chunks = self.split_file(file, max_chunk_size, file_language)

            for chunk in chunks:
                all_chunks.append(
                    {
                        "path": str(file),
                        "relative_path": str(file.relative_to(self._repo_path)),
                        "file_type": file_type,
                        "language": file_language,
                        "chunk_id": i,
                        "code": chunk[0],
                        "start_line": chunk[1],
                        "end_line": chunk[2],
                    }
                )
                i += 1
        return all_chunks


if __name__ == "__main__":
    repo_path = Path("data/repos/escrcpy")
    code_parser = CodeParser(repo_path)
    chunks = code_parser.parse(30)
    pprint(len(chunks))
