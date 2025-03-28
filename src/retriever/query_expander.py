import numpy as np
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.retriever.candidate_terms import candidate_terms


class QueryExpander:
    @staticmethod
    def expand_query_with_embeddings(
        query: str, model: SentenceTransformer, top_k: int = 10
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
