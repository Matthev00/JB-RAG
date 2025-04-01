import json
from src.config import USE_OPENAI

if USE_OPENAI:
    import os
    from dotenv import load_dotenv
    from openai import AzureOpenAI

    load_dotenv()

    def generate_summary(query: str, results: list[dict]) -> str:
        """
        Generate a short explanation of search results using Azure OpenAI.

        Args:
            query (str): The user's query.
            results (list[dict]): List of matched code snippets with metadata.

        Returns:
            str: Generated explanation text.
        """
        metadata = json.dumps(
            [
                {
                    "content": r["code"],
                    "path": r["relative_path"],
                    "file_type": r["file_type"],
                    "language": r["language"],
                    "start_line": r["start_line"],
                    "end_line": r["end_line"],
                }
                for r in results
            ],
            indent=2,
        )

        system_msg = "You are a helpful assistant for a code search engine."
        user_msg = f"""User asked: "{query}"

            The following files were retrieved:
            {metadata}

            Briefly explain:
            1. What the query is about.
            2. What kind of files were returned and why they are relevant.
            Keep the explanation short (max 2–3 sentences per retrieved file)."""

        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2023-03-15-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=500,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

else:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from src.config import DEVICE, SUMMARY_MODEL

    def format_prompt(query: str, results: list[dict]) -> str:
        """
        Format the input prompt for LLM-based summarization.

        Args:
            query (str): User's query.
            results (list[dict]): List of code snippets and metadata.

        Returns:
            str: Prompt formatted for causal language model.
        """
        metadata = json.dumps(
            [
                {
                    "content": r["code"],
                    "file_type": r["file_type"],
                    "language": r["language"],
                }
                for r in results
            ],
            indent=2,
        )

        return f"""<|system|>
            You are a helpful assistant for a code search engine.</s>
            <|user|>
            User asked: "{query}"

            The following files were retrieved:
            {metadata}

            Briefly explain:
            1. What the query is about.
            2. What kind of files were returned and why they are relevant.
            </s>
            <|assistant|>
            """

    def load_model():
        """
        Load a local summarization model and tokenizer.

        Returns:
            transformers.Pipeline: A text generation pipeline.
        """
        model = AutoModelForCausalLM.from_pretrained(
            SUMMARY_MODEL,
            device_map=DEVICE,
            torch_dtype=torch.float32,
        )
        tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL)
        return pipeline("text-generation", model=model, tokenizer=tokenizer)

    def generate_summary(query: str, results: list[dict]) -> str:
        """
        Generate a short explanation of search results using a local LLM.

        Args:
            query (str): The user's query.
            results (list[dict]): List of matched code snippets with metadata.

        Returns:
            str: Generated explanation text.
        """
        prompt = format_prompt(query, results)
        model = load_model()

        response = model(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.5,
            truncation=True
        )[0]["generated_text"]

        return response[len(prompt):].strip()
