import json
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

def load_dataset(jsonl_path: Path, limit: int = 10) -> List[dict]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for i, line in enumerate(f) if i < limit]

def build_prompt(prompt_text: str, input_group: list[dict]) -> str:
    instruction = f"### Instruction:\n{prompt_text.strip()}\n\n"
    code_fragments = ""
    for snippet in input_group:
        rel_path = snippet.get("relative_path", "unknown.c")
        code = snippet["code"]
        code_fragments += f"--- File: {rel_path} ---\n{code.strip()}\n\n"

    retrieved = f"RETRIEVED FRAGMENTS:\n{code_fragments.strip()}\n\n"
    return instruction + retrieved + "### Response:\n"


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    eos = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("<|endoftext|>")
    pad = tokenizer.pad_token_id or eos

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=pad,
            eos_token_id=eos,
        )

    generated_tokens = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def load_base_model(name: str):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")
    return model, tokenizer

def load_lora_model(base_name: str, lora_path: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, lora_path)
    return model, tokenizer

jsonl_path = Path("/content/drive/MyDrive/JB-RAG/data/dataset.jsonl")
base_model_name = "TechxGenus/starcoder2-3b-instruct"
lora_model_path = Path("/content/drive/MyDrive/JB-RAG/models/instruct/checkpoint-102/")

base_model, base_tokenizer = load_base_model(base_model_name)
lora_model, lora_tokenizer = load_lora_model(base_model_name, lora_model_path)

entries = load_dataset(jsonl_path, limit=3)

for i, entry in enumerate(tqdm(entries)):
    prompt = build_prompt(entry["synthetic_prompt"], entry["input_group"])

    base_output = generate(base_model, base_tokenizer, prompt, 1024)
    lora_output = generate(lora_model, lora_tokenizer, prompt, 1024)
    ground_truth = entry["generated_summary"]

    print(f"\n========== Example {i+1} ==========")
    print(">> PROMPT:\n", entry["synthetic_prompt"])
    print(">> BASE OUTPUT:\n", base_output.strip())
    print(">> LORA OUTPUT:\n", lora_output.strip())
    print(">> GROUND TRUTH:\n", ground_truth.strip())
