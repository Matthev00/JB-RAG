import pandas as pd
import random
import json
from datasets import load_dataset
from pathlib import Path

class DatasetExtractr:
    def __init__(self, dataset_name, split="vaidation"):
        """
        Initializes the DatasetExtractor with the dataset name and split.
        
        Args:
            dataset_name (str): Name of the dataset to load.
            split (str): The split of the dataset to load (default is "train_small").
        """
        self.dataset_name = dataset_name
        self.split = split
        self.df: pd.DataFrame = None
        self.sampled_df = None
        self.dataset = []

    def load_dataset(self):
        """
        Downloads the dataset using the Hugging Face datasets library.
        """
        dataset = load_dataset(self.dataset_name, split=self.split)
        self.df = dataset.to_pandas()

    def preprocess(self):
        """
        Preprocesses the dataset by selecting relevant columns and sampling.
        """
        self.df = self.df[["code", "docstring", "language", "path"]]

        self.sampled_df = (
            self.df.groupby("language")
            .apply(lambda x: x.sample(n=min(len(x), 6000), random_state=42))
            .reset_index(drop=True)
        )

    def create_dataset(self):
        """
        Creates a dataset by iterating over the sampled DataFrame and generating input-output pairs.
        """
        for _, row in self.sampled_df.iterrows():
            start_line = random.randint(1, 30)
            code_lines = row["code"].count("\n") + 1
            end_line = start_line + code_lines

            input_dict = {
                "relative_path": row["path"],
                "file_type": "code",
                "language": row["language"],
                "code": row["code"],
                "start_line": start_line,
                "end_line": end_line,
            }
            output = row["docstring"]
            self.dataset.append({"input": input_dict, "output": output})

    def save_dataset(self, output_path: Path):
        """
        Saves the dataset to a JSONL file.

        Args:
            output_path (Path): Path to the output file.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for item in self.dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def pipline(self, output_path: Path):
        """
        Runs the entire pipeline: load, preprocess, create dataset, and save.
        """
        self.load_dataset()
        self.preprocess()
        self.create_dataset()
        self.save_dataset(output_path)


if __name__ == "__main__":
    processor = DatasetExtractr(dataset_name="Fsoft-AIC/the-vault-function", split="train_small")
    processor.pipline(Path("data/summary/processed/dataset.jsonl"))
