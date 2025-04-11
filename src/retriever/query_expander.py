import numpy as np
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.retriever.candidate_terms import candidate_terms


class QueryExpander:
    _llm_model = None

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

    @classmethod
    def _load_llm(cls):
        """
        Loads the large language model (LLM) for query expansion.
        This method checks if the model is already loaded, and if not, it initializes it.
        """
        if cls._llm_model is None:
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                pipeline,
                BitsAndBytesConfig,
            )
            from src.config import QUERY_EXPANDER_MODEL, DEVICE

            tokenizer = AutoTokenizer.from_pretrained(
                QUERY_EXPANDER_MODEL, trust_remote_code=True
            )

            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                QUERY_EXPANDER_MODEL,
                quantization_config=bnb_config,
                trust_remote_code=True,
                device_map=DEVICE,
            )
            cls._llm_model = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=DEVICE,
            )

    @classmethod
    def generate_query_expansion(cls, query: str) -> str:
        """
        Generates an expanded query using a LLM.

        Args:
            query (str): Original query.

        Returns:
            str: Rewrited query.
        """
        cls._load_llm()

        system_prompt = "You are a helpful query expansion assistant. Your task is to rewrite and expand the user's query for better code search."
        user_prompt = f"User query: {query}\nRewrite and expand this query for better code search. Be technical and precise."
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>"

        output = cls._llm_model(
            prompt, max_new_tokens=128, do_sample=True, temperature=0.7
        )[0]["generated_text"]
        return output.split("<|assistant|>")[1].split("<|end|>")[0].strip()

    @classmethod
    def generate_code_snippet(cls, query: str, language: str = "JavaScript") -> str:
        """
        Generates a code snippet using a LLM.

        Args:
            query (str): Original query.
            language (str): Programming language for the code snippet.

        Returns:
            str: Generated code snippet.
        """
        cls._load_llm()

        system_prompt = "You are a helpful coding assistant. Always return code only, with no explanation."
        user_prompt = f"Generate a relevant {language} code snippet that solves the following problem:\n{query}"
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>"

        output = cls._llm_model(
            prompt, max_new_tokens=128, do_sample=True, temperature=0.7
        )[0]["generated_text"]
        return output.split("<|assistant|>")[1].split("<|end|>")[0].strip()

    @staticmethod
    def query_with_LLM(query: str) -> str:
        """
        Rewrites the query using a large language model (LLM).
        Use LLM to generate code that could be similar to the original query.

        Args:
            query (str): Original query.

        Returns:
            str: Expanded query.
        """

        return query


qe = QueryExpander()

query = "How does the repository handle IPv6 addresses in ADB commands?"
expanded = qe.generate_query_expansion(query)
code = qe.generate_code_snippet(query, language="JavaScript")

print("Expanded query:\n", expanded)
print("\nGenerated code:\n", code)
