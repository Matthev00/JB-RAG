REPO_DIR = "data/repos"

DEFAULT_IGNORED_EXTENSIONS = {
    ".jpeg",
    ".jpg",
    ".png",
    ".gif",
    ".svg",
}
DOC_EXTENSIONS = {".md", ".pdf"}
CONFIG_EXTENSIONS = {".json", ".yml", ".yaml", ".env", ".toml", ".ini", ".xml"}


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
