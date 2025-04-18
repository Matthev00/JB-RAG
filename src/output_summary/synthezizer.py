import json
import random
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from together import Together
from tqdm import tqdm


class LLMProcessor:
    def __init__(self, model: str):
        """
        Inicialize the LLMProcessor with the specified model.

        Args:
            model (str): Model name to be used with Together API.
        """
        self.client = Together()
        self.model = model

    def generate_summary(self, input_data: list[dict], output_data: list[str]) -> str:
        """
        Generates a summary for the provided code and discription using Together API.
        This method constructs a prompt using the code and metadata, and then
        sends it to the Together API for processing. The API response is then
        returned as the summary.
        The input_data should contain the code and metadata, while output_data
        should contain discription of each code in the input_data.

        Args:
            input_data (list[dict]): Input data containing code and metadata.
            output_data (list[str]): Output data containing discription of each code.

        Returns:
            str: Generated summary.

        """
        system_prompt = (
            "You are a highly skilled assistant specialized in generating concise, accurate, and professional summaries "
            "for groups of code snippets. Your goal is to analyze the, metadata with code, and existing descriptions, "
            "and produce a single summary that captures the purpose and functionality of the group in a clear and precise manner. "
            "Follow these guidelines:\n"
            "- Focus on the main functionality of the code snippets as a group.\n"
            "- Use technical language appropriate for developers.\n"
            "- Avoid unnecessary details or redundant information.\n"
            "- Ensure the whole summary is no longer than 5-7 sentences.\n"
            "- If the code or descriptions are incomplete or unclear, make reasonable assumptions and note them in the summary."
        )
        user_prompt = "Analyze the following group of code snippets, their metadata, and existing descriptions. Then generate a concise and accurate summary, Return only summary:\n\n"
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
        user_prompt += "Summary: "

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
        while i < len(records):
            group_size = random.randint(1, max_group_size)
            group = records[i : i + group_size]
            i += group_size

            input_data = [record["input"] for record in group]
            output_data = [record["output"] for record in group]

            new_output = llm_processor.generate_summary(input_data, output_data)

            processed_dataset.append(
                {"input_group": input_data, "generated_summary": new_output}
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in processed_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    load_dotenv()
    llm_processor = LLMProcessor(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
    input_file = Path("data/summary/processed/dataset.jsonl")
    output_file = Path("data/summary/synthetic/dataset.jsonl")
    process_dataset(input_file, output_file, llm_processor)
