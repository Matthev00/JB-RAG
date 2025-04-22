import json
import random
from collections import defaultdict
from pathlib import Path

from together import Together
from tqdm import tqdm, trange


class LLMProcessor:
    def __init__(self, model: str):
        """
        Inicialize the LLMProcessor with the specified model.

        Args:
            model (str): Model name to be used with Together API.
        """
        self.client = Together()
        self.model = model

    def generate_rag_prompt(
        self, input_data: list[dict], output_data: list[str]
    ) -> str:
        """
        Generates a synthetic user prompt that would likely retrieve this group of code snippets
        in a RAG system based on their content and descriptions.

        Args:
            input_data (list[dict]): Input code snippets and metadata.
            output_data (list[str]): Existing descriptions of the code snippets.

        Returns:
            str: A synthetic user prompt suitable for use with a RAG retriever.
        """
        system_prompt = (
            "You are simulating a user interacting with a retrieval-based code assistant. "
            "Given a group of code snippets and their descriptions, generate a realistic question "
            "the user might ask that would cause the system to retrieve this specific group of snippets. "
            "Do not refer to specific variables or filenames — just ask a question that matches their functionality."
            "Return only the question user might ask, without any additional text."
        )

        user_prompt = "Here is a group of code snippets with descriptions:\n\n"

        for i, (input_item, output_item) in enumerate(
            zip(input_data, output_data), start=1
        ):
            user_prompt += (
                f"Snippet {i}:\n"
                f"{input_item['code']}\n\n"
                f"Metadata:\n"
                f"- Language: {input_item['language']}\n"
                f"- Start line: {input_item['start_line']}\n"
                f"- End line: {input_item['end_line']}\n"
                f"Description: {output_item}\n"
                f"{'-' * 40}\n"
            )

        user_prompt += "What is a realistic user question that would likely retrieve this group of code snippets?"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error during prompt generation: {e}")
            return "How it is done in the code?"

    def generate_summary(
        self, synthetic_prompt: str, input_data: list[dict], output_data: list[str]
    ) -> str:
        """
        Generates a summary for the provided code and description using Together API.
        This method constructs a prompt using the code, metadata, user prompt, and descriptions,
        then sends it to the Together API for processing.

        Args:
            synthetic_prompt (str): The prompt to be used for generating the summary.
            input_data (list[dict]): Input data containing code and metadata.
            output_data (list[str]): Output data containing description of each code.

        Returns:
            str: Generated summary.
        """
        system_prompt = (
            "You are a highly skilled assistant specialized in generating concise, accurate, and professional summaries "
            "for questions and groups of code snippets. Your goal is to analyze the provided code, metadata, and existing descriptions, "
            "and produce a summary that captures the purpose and functionality of the group in a clear and precise manner. "
            "If the code snippets are unrelated or serve different purposes, describe each snippet individually instead of "
            "trying to combine them into a single summary. Follow these guidelines:\n"
            "- Focus on the main functionality of the code snippets.\n"
            "- Use technical language appropriate for developers.\n"
            "- Avoid unnecessary details or redundant information.\n"
            "- Ensure the summary is no longer than 5-7 sentences.\n"
            "- If the code or descriptions are incomplete or unclear, make reasonable assumptions and note them in the summary."
        )

        user_prompt = f"User question: {synthetic_prompt}\n\n"
        user_prompt += "Analyze the following group of code snippets, their metadata, and existing descriptions. Then generate concise and accurate summaries. If the snippets are unrelated, describe each snippet individually:\n\n"

        for i, (input_item, output_item) in enumerate(
            zip(input_data, output_data), start=1
        ):
            user_prompt += (
                f"Snippet {i}:\n"
                f"{input_item['code']}\n\n"
                f"Metadata:\n"
                f"- File Path: {input_item['relative_path']}\n"
                f"- Language: {input_item['language']}\n"
                f"- Start line: {input_item['start_line']}\n"
                f"- End line: {input_item['end_line']}\n"
                f"Existing Description: {output_item}\n"
                f"{'=' * 40}\n"
            )
        user_prompt += "Summary:"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error during API call: {e}")
            return "Error: Unable to generate summary."


def rephrase_how_to_where(question: str) -> str:
    """
    Replaces 'How can I' at the beginning of a question with 'Where can I'.

    Args:
        question (str): The original question.

    Returns:
        str: Rephrased question.
    """
    if question.lower().startswith("how can i"):
        return "Where can I" + question[9:]
    return question


def process_dataset(
    input_file: Path,
    output_file: Path,
    llm_processor: LLMProcessor,
    max_group_size: int = 6,
) -> None:
    """
    Processes the dataset by generating summaries for grouped records using the
    specified LLMProcessor. The input dataset is read from the input_file,
    and the processed dataset is saved to the output_file.

    Args:
        input_file (Path): Path to the input dataset file.
        output_file (Path): Path to save the processed dataset.
        llm_processor (LLMProcessor): Instance of LLMProcessor for generating summaries.
        max_group_size (int): Max number of records to group together for summary generation.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    grouped_by_language = defaultdict(list)
    for record in dataset:
        grouped_by_language[record["input"]["language"]].append(record)

    processed_dataset = []

    for language, records in tqdm(grouped_by_language.items()):
        i = 0
        with trange(len(records), desc=f"Processing {language}") as pbar:
            while i < len(records):
                group_size = random.randint(1, max_group_size)
                group = records[i : i + group_size]

                input_data = [record["input"] for record in group]
                output_data = [record["output"] for record in group]

                synthetic_prompt = llm_processor.generate_rag_prompt(
                    input_data, output_data
                )
                synthetic_prompt = rephrase_how_to_where(synthetic_prompt)
                new_output = llm_processor.generate_summary(
                    synthetic_prompt, input_data, output_data
                )

                processed_dataset.append(
                    {
                        "synthetic_prompt": synthetic_prompt,
                        "input_group": input_data,
                        "generated_summary": new_output,
                    }
                )

                i += group_size
                pbar.update(group_size)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in processed_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
