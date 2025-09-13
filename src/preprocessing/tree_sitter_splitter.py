from pathlib import Path

from src.preprocessing.tree_sitter_setup import TreeSitterManager


def _get_named_descendants(node):
    for child in node.children:
        if child.is_named:
            yield child
            yield from _get_named_descendants(child)


class TreeSitterSplitter:
    def __init__(self):
        self.tree_sitter_manager = TreeSitterManager()

    def is_supported(self, language: str) -> bool:
        """
        Checks if the given language is supported by Tree-sitter.

        Args:
            language (str): Programming language to check.

        Returns:
            bool: True if supported, False otherwise.
        """
        return language in self.tree_sitter_manager.language_map.keys()

    def split(
        self, file_path: Path, language: str, max_chunk_size: int = 100
    ) -> list[tuple[str, int, int]]:
        """
        Splits the given file using Tree-sitter into semantically meaningful chunks.

        Args:
            file_path (Path): Path to the file to be split.
            language (str): Programming language of the file.

        Returns:
            List of tuples: (code_chunk_str, start_line, end_line)
        """
        try:
            parser = self.tree_sitter_manager.get_parser(language)
        except Exception as e:
            print(f"Error loading parser for {language}: {file_path}")
            return []

        source_code = file_path.read_text(encoding="utf-8", errors="ignore")
        tree = parser.parse(bytes(source_code, "utf-8"))
        root = tree.root_node

        lines = source_code.splitlines(keepends=True)
        chunks = []

        for node in _get_named_descendants(root):
            if node.type in {
                "function_definition",
                "function",
                "method_definition",
                "class_definition",
                "class",
            }:
                start, end = node.start_point[0], node.end_point[0]
                if end >= len(lines):
                    end = len(lines) - 1
                while start <= end:
                    chunk_end = min(start + max_chunk_size - 1, end)
                    chunk = "".join(lines[start : chunk_end + 1])
                    chunks.append((chunk, start, chunk_end))
                    start = chunk_end + 1

        return chunks
