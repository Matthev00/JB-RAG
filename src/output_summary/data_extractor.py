import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset


class DatasetExtractor:
    """
    A class to extract, preprocess, and save datasets in JSONL format.

    Attributes:
        dataset_name (str): Name of the dataset to process.
        split (str): Dataset split to process (e.g., "train", "test").
        df (pd.DataFrame): Loaded dataset as a pandas DataFrame.
        dataset (List[Dict]): Processed dataset stored in memory.
    """

    def __init__(self, dataset_name: str, split: str = "validation"):
        """
        Initializes the DatasetExtractor with the dataset name and split.

        Args:
            dataset_name (str): Name of the dataset to load.
            split (str): The split of the dataset to load (default is "validation").
        """
        self.dataset_name = dataset_name
        self.split = split
        self.df: pd.DataFrame = None
        self.dataset: list[dict] = []

    def load_dataset(self):
        """
        Downloads the dataset using the Hugging Face datasets library and converts it to a pandas DataFrame.
        """
        dataset = load_dataset(self.dataset_name, split=self.split)
        self.df = dataset.to_pandas()

    def preprocess(self):
        """
        Preprocesses the dataset by selecting relevant columns and grouping by language and repository.
        """
        self.df = self.df[["code", "docstring", "language", "path", "repo"]]
        grouped = self.df.groupby(["language", "repo"])
        self.dataset = self.create_dataset(grouped)

    def sample_group(self, group: pd.DataFrame, max_samples: int = 6) -> list[dict]:
        """
        Samples a limited number of records from a group.

        Args:
            group (pd.DataFrame): Group of records to sample from.
            max_samples (int): Maximum number of samples to take from the group.

        Returns:
            List[Dict]: A list of sampled records.
        """
        sampled_group = group.sample(n=min(len(group), max_samples), random_state=42)
        sampled_records = []

        for _, row in sampled_group.iterrows():
            start_line = random.randint(1, 30)
            code_lines = row["code"].count("\n") + 1
            end_line = start_line + code_lines

            input_dict = {
                "relative_path": row["path"],
                "file_type": "code",
                "language": row["language"],
                "repo": row["repo"],
                "code": row["code"],
                "start_line": start_line,
                "end_line": end_line,
            }
            output = row["docstring"]

            sampled_records.append({"input": input_dict, "output": output})

        return sampled_records

    def group_by_repo(self, records: list[dict]) -> dict:
        """
        Groups records by their repository.

        Args:
            records (List[Dict]): List of records to group.

        Returns:
            Dict[str, List[Dict]]: A dictionary where keys are repository names and values are lists of records.
        """
        repo_groups = defaultdict(list)
        for record in records:
            repo = record["input"]["repo"]
            repo_groups[repo].append(record)
        return repo_groups

    def limit_samples_per_language(
        self, dataset: dict, max_samples: int = 600
    ) -> list[dict]:
        """
        Limits the number of samples per language to a maximum value.

        Args:
            dataset (Dict[str, List[Dict]]): Dataset grouped by language.
            max_samples (int): Maximum number of samples per language.

        Returns:
            List[Dict]: A list of records with limited samples per language.
        """
        final_dataset = []

        for language, records in dataset.items():
            repo_groups = self.group_by_repo(records)

            sampled_records = []
            for repo, repo_records in repo_groups.items():
                sampled_records.extend(repo_records)
                if len(sampled_records) > max_samples:
                    break

            final_dataset.extend(sampled_records)

        return final_dataset

    def create_dataset(self, grouped: pd.DataFrame) -> list[dict]:
        """
        Creates a dataset by iterating over grouped data and generating input-output pairs.

        Args:
            grouped (pd.DataFrame): Grouped DataFrame by language and repository.

        Returns:
            List[Dict]: A list of processed records.
        """
        dataset = defaultdict(list)

        for (language, repo), group in grouped:
            sampled_records = self.sample_group(group)
            dataset[language].extend(sampled_records)

        final_dataset = self.limit_samples_per_language(dataset)
        return final_dataset

    def save_dataset(self, output_path: Path):
        """
        Saves the processed dataset to a JSONL file.

        Args:
            output_path (Path): Path to the output file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in self.dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def pipeline(self, output_path: Path):
        """
        Runs the entire pipeline: load, preprocess, create dataset, and save.

        Args:
            output_path (Path): Path to the output file.
        """
        self.load_dataset()
        self.preprocess()
        self.save_dataset(output_path)


if __name__ == "__main__":
    processor = DatasetExtractor(
        dataset_name="Fsoft-AIC/the-vault-function", split="train_small"
    )
    processor.pipeline(Path("data/summary/processed/dataset.jsonl"))
