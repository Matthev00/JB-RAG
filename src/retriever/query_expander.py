import numpy as np
from dotenv import load_dotenv
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from together import Together

from src.config import QUERY_EXPANDER_MODEL
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
                    "content": "You are a helpful assistant, who writes code. Always return only the code snippet nothing else.",
                },
                {
                    "role": "user",
                    "content": f"Generate a relevant {language} code snippet for the following query: {query}. Return only the code snippet nothing else.",
                },
            ],
        )

        return response.choices[0].message.content

    @staticmethod
    def query_with_LLM(
        query: str, model: SentenceTransformer, language: str = "JavaScript"
    ) -> np.ndarray:
        """
        Expands the query using Together AI, generates a code snippet, computes embeddings for both,
        and returns a weighted combination.

        Args:
            query (str): Original user query.
            model (SentenceTransformer): Model used to embed text/code.
            language (str): Programming language for code generation.

        Returns:
            np.ndarray: Combined embedding vector.
        """
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
    query = "How to use FAISS for similarity search?"
    expanded_query = QueryExpander.query_with_together_api(query, model)

    print("Expanded Query:", expanded_query)
