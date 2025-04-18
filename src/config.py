REPO_DIR = "data/repos"
EMBEDDINGS_DIR = "data/embeddings"
FAISS_INDEX_DIR = "data/faiss"
MAX_CHUNK_SIZE = 110

REPO_URL = "https://github.com/viarotel-org/escrcpy.git"

DEFAULT_IGNORED_EXTENSIONS = {
    ".jpeg",
    ".jpg",
    ".png",
    ".gif",
    ".svg",
}
DOC_EXTENSIONS = {".md", ".pdf"}
CONFIG_EXTENSIONS = {".json", ".yml", ".yaml", ".env", ".toml", ".ini", ".xml"}
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".vue",
    ".cpp",
    ".java",
    ".html",
    ".css",
    ".sh",
    ".sql",
    ".r",
    ".rb",
    ".php",
    ".cs",
    ".go",
    ".yaml",
    ".json",
    ".xml",
}


LANGUAGE_PATTERNS = {
    "Python": r"^\s*(def |class |async def |from |if |elif |else |try |except |with |for |while )",
    "JavaScript": r"^\s*(function |const |let |var |class |if |else |for |while |do |switch |case )",
    "TypeScript": r"^\s*(function |const |let |var |class |interface |type |enum |if |else |for |while |do |switch |case )",
    "Vue": r"<script>|<style>|<template>|<component>|<router-view>|<slot>",
    "C++": r"^\s*(class |struct |namespace |template |void |int |float |double |char |bool |if |else |for |while |do |switch |case |#define )",
    "Java": r"^\s*(public |private |protected |class |interface |enum |void |int |float |double |char |boolean |if |else |for |while |do |switch |case |try |catch |finally )",
    "HTML": r"<script>|<style>|<div>|<span>|<p>|<a>|<ul>|<ol>|<li>|<table>|<tr>|<td>|<th>|<header>|<footer>|<section>|<article>|<nav>|<form>|<input>|<button>|<textarea>|<select>|<option>|<h[1-6]>",
    "Unknown": r".*",
}


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SUMMARY_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cpu"
USE_OPENAI = True
