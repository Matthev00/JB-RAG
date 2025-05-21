from pathlib import Path

from dotenv import load_dotenv

from src.config import SUMMARY_DATASET_MODEL, SUMMARY_DIR
from src.output_summary.data_extractor import DatasetExtractor
from src.output_summary.synthesizer import LLMProcessor, process_dataset


def main():
    """Main function to create a summary dataset."""
    load_dotenv()
    processor = DatasetExtractor(
        dataset_name="Fsoft-AIC/the-vault-function", split="train_small"
    )
    processed_data_path = Path(SUMMARY_DIR) / "processed" / "dataset.jsonl"
    processor.pipeline(output_path=processed_data_path)

    llm_processor = LLMProcessor(model=SUMMARY_DATASET_MODEL)
    synthetic_data_path = Path(SUMMARY_DIR) / "synthetic" / "dataset.jsonl"
    process_dataset(processed_data_path, synthetic_data_path, llm_processor)


if __name__ == "__main__":
    main()
