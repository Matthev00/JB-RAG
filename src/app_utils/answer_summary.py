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
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
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
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig

    from src.config import DEVICE, SUMMARY_MODEL, LORA_PATH

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
    
    def load_model_with_optional_lora(
        lora_path: str | None = None,
    ):
        if DEVICE == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

            model = AutoModelForCausalLM.from_pretrained(
                SUMMARY_MODEL,
                quantization_config=bnb_config,
                device_map=DEVICE,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                SUMMARY_MODEL,
                device_map=DEVICE,
                torch_dtype=torch.float32,
            )

        if lora_path:
            model = PeftModel.from_pretrained(model, lora_path)
            print(f"Loaded LoRA adapter from {lora_path}")
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL)
        return model, tokenizer

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
    
    def generate_response(prompt: str, model, tokenizer, max_new_tokens: int = 150) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.5,
            pad_token_id=tokenizer.eos_token_id
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

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
        model, tokenizer = load_model_with_optional_lora(LORA_PATH)

        # response = model(
        #     prompt, max_new_tokens=150, do_sample=True, temperature=0.5, truncation=True
        # )[0]["generated_text"]
        response = generate_response(prompt, model, tokenizer)

        return response[len(prompt) :].strip()
