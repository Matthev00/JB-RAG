from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from together import Together

from src.config import PROJECT_NAME, QUERY_EXPANDER_MODEL, REPO_DIR
from src.preprocessing.code_parser import CodeParser
from src.retriever.candidate_terms import candidate_terms

load_dotenv()


class QueryExpander:
    @staticmethod
    def expand_query_with_embeddings(
        query: str, model: SentenceTransformer, top_k: int = 5
    ) -> str:
        """
        Expands the query by finding semantically similar terms using embeddings.

        Args:
            query (str): Original query.
            model (SentenceTransformer): Pretrained SentenceTransformer model.
            top_k (int): Number of similar terms to add.

        Returns:
            str: Expanded query.
        """

        query_embedding = model.encode([query], convert_to_numpy=True)
        candidate_embeddings = model.encode(candidate_terms, convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

        top_indices = np.argsort(similarities)[-top_k:]
        expanded_terms = [candidate_terms[i] for i in top_indices]

        return query + " " + " ".join(expanded_terms)

    @staticmethod
    def expand_query_with_wordnet(query: str) -> str:
        """
        Expands the query using WordNet to find synonyms.

        Args:
            query (str): Original query.

        Returns:
            str: Expanded query.
        """
        expanded_terms = []
        for word in query.split():
            synonyms = wordnet.synsets(word)
            for syn in synonyms:
                for lemma in syn.lemmas():
                    expanded_terms.append(lemma.name())

        return query + " " + " ".join(set(expanded_terms))

    @staticmethod
    def expand_query_with_together_api(query: str) -> str:
        """
        Expands the query using Together AI API.

        Args:
            query (str): Original query.

        Returns:
            str: Expanded query.
        """
        load_dotenv()
        client = Together()
        response = client.chat.completions.create(
            model=QUERY_EXPANDER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant, who helps to expand queries for better search results. Always return only the expanded query.",
                },
                {
                    "role": "user",
                    "content": f"Rewrite the query to make it more technical and precise: {query}. Return only new query nothing else.",
                },
            ],
        )

        return response.choices[0].message.content

    @staticmethod
    def generate_code_snippet_with_together_api(
        query: str, language: str = "JavaScript"
    ) -> str:
        """
        Generates a code snippet using Together AI API.

        Args:
            query (str): Expanded query.
            language (str): Programming language for the code snippet.

        Returns:
            str: Generated code snippet.
        """
        load_dotenv()
        client = Together()
        response = client.chat.completions.create(
            model=QUERY_EXPANDER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant, who writes code. Always return only the code snippet nothing else. Keep response short.",
                },
                {
                    "role": "user",
                    "content": f"Generate a relevant code snippet in {language} for the following query: {query}. Return only the code snippet nothing else.",
                },
            ],
        )

        return response.choices[0].message.content

    @staticmethod
    def detect_language_from_files() -> str:
        """
        Detects the programming languages from a list of files.
        Returns the most common language(s), and includes others if their frequency
        is within 10% of the most common one.

        Returns:
            str: Detected programming language(s), e.g. "Python", or "Python and Java".
        """
        code_parser = CodeParser(Path(REPO_DIR) / PROJECT_NAME)
        files = code_parser.get_relevant_files()
        languages = []
        for file in files:
            language = code_parser._detect_language(file)
            if language:
                languages.append(language)

        if not languages:
            return "JavaScript"

        counter = Counter(languages)
        most_common = counter.most_common()
        top_count = most_common[0][1]

        close_languages = [
            lang
            for lang, count in most_common
            if count >= 0.5 * top_count and lang != "Unknown"
        ]

        return " and ".join(close_languages)

    @staticmethod
    def query_with_LLM(query: str, model: SentenceTransformer) -> np.ndarray:
        """
        Expands the query using Together AI, generates a code snippet, computes embeddings for both,
        and returns a weighted combination.

        Args:
            query (str): Original user query.
            model (SentenceTransformer): Model used to embed text/code.

        Returns:
            np.ndarray: Combined embedding vector.
        """
        language = QueryExpander.detect_language_from_files([Path(PROJECT_NAME)])

        expanded_query = QueryExpander.expand_query_with_together_api(query)
        generated_code = QueryExpander.generate_code_snippet_with_together_api(
            expanded_query, language
        )

        query_embed = model.encode([expanded_query], convert_to_numpy=True)
        code_embed = model.encode([generated_code], convert_to_numpy=True)

        combined_embedding = 0.6 * query_embed + 0.4 * code_embed
        return combined_embedding


if __name__ == "__main__":
    # Example usage
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query = (
        "What functionality does the component provide for mirroring a device's screen?"
    )
    expanded_query = QueryExpander.detect_language_from_files()

    print("Expanded Query:", expanded_query)
