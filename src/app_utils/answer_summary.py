import json

from src.config import USE_OPENAI

if USE_OPENAI:
    import os

    import openai
    from dotenv import load_dotenv

    from src.config import OPENAI_MODEL

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate_summary(query: str, results: list[dict]) -> str:
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
            Keep the explanation short (max 2 sentences per retrieved file)."""

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
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
        model = AutoModelForCausalLM.from_pretrained(
            SUMMARY_MODEL,
            device_map=DEVICE,
            torch_dtype=torch.float32,
        )
        tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL)
        return pipeline("text-generation", model=model, tokenizer=tokenizer)

    def generate_summary(query: str, results: list[dict]) -> str:
        prompt = format_prompt(query, results)
        model = load_model()
        response = model(prompt, max_new_tokens=150, do_sample=True, temperature=0.5)[
            0
        ]["generated_text"]
        answer = response[len(prompt) :].strip()
        return answer


if __name__ == "__main__":
    results = [
        {
            "path": "/home/mateusz/JB-RAG/data/repos/escrcpy/electron/resources/extra/mac/android-platform-tools/NOTICE.txt",
            "relative_path": "electron/resources/extra/mac/android-platform-tools/NOTICE.txt",
            "file_type": "other",
            "language": "Unknown",
            "chunk_id": 329,
            "code": "    in the event an application does not supply such function or\n",
            "start_line": 7699,
            "end_line": 7809,
        },
    ]

    output = generate_summary("whats up?", results)
    print(50 * "-")
    print(output)
