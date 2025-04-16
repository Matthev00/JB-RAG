import os
from pathlib import Path

from tree_sitter import Language, Parser

LANGUAGE_REPOS = {
    "python": "https://github.com/tree-sitter/tree-sitter-python",
    "javascript": "https://github.com/tree-sitter/tree-sitter-javascript",
    "java": "https://github.com/tree-sitter/tree-sitter-java",
    "c_sharp": "https://github.com/tree-sitter/tree-sitter-c-sharp",
    "rust": "https://github.com/tree-sitter/tree-sitter-rust",
    "html": "https://github.com/tree-sitter/tree-sitter-html",
    "css": "https://github.com/tree-sitter/tree-sitter-css",
}


class TreeSitterBuilder:
    def __init__(
        self, vendor_dir="data/vendors", lib_path="data/build/my-languages.so"
    ):
        self.vendor_dir = Path(vendor_dir)
        self.lib_path = Path(lib_path)
        self.vendor_dir.mkdir(parents=True, exist_ok=True)

    def _get_parser_dir(self, lang: str) -> Path:
        """
        Returns the directory where the parser for the specified language is located.

        Args:
            lang (str): Programming language for which to get the parser directory.

        Returns:
            path to the parser directory.
        """
        if lang == "typescript":
            return self.vendor_dir / "tree-sitter-typescript" / "tsx"
        return self.vendor_dir / f"tree-sitter-{lang}"

    def clone_repos(self):
        """
        Clones the Tree-sitter repositories for the specified languages.
        """
        for lang, repo in LANGUAGE_REPOS.items():
            target_path = self._get_parser_dir(lang)
            if not target_path.exists():
                os.system(
                    f"git clone {repo} {target_path.parent if lang == 'typescript' else target_path}"
                )

    def build_library(self):
        """
        Builds the Tree-sitter library by compiling the parsers.
        """
        parser_dirs = [self._get_parser_dir(lang) for lang in LANGUAGE_REPOS]
        self.lib_path.parent.mkdir(parents=True, exist_ok=True)
        Language.build_library(
            str(self.lib_path), [str(p) for p in parser_dirs if p.exists()]
        )

    def setup(self):
        """
        Sets up the Tree-sitter library by cloning repositories and building the library.
        """
        if not self.lib_path.exists():
            self.clone_repos()
            self.build_library()


class TreeSitterManager:
    def __init__(self, lib_path="data/build/my-languages.so"):
        """
        Initializes the TreeSitterManager with the path to the compiled library.

        Args:
            lib_path (str): Path to the compiled Tree-sitter library.
        """
        self.lib_path = lib_path
        self.language_map = {
            "Python": "python",
            "JavaScript": "javascript",
            "Java": "java",
            "C#": "c_sharp",
            "Rust": "rust",
            "HTML": "html",
            "CSS": "css",
        }
        self.languages = {
            name: Language(self.lib_path, parser_name)
            for name, parser_name in self.language_map.items()
        }

    def get_parser(self, language: str) -> Parser | None:
        """
        Returns a Tree-sitter parser for the specified language.

        Args:
            language (str): Programming language for which to get the parser.

        Returns:
            Parser | None: Tree-sitter parser for the specified language, or None if not supported.
        """
        if language not in self.languages:
            return None
        parser = Parser()
        parser.set_language(self.languages[language])
        return parser
