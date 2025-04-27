import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import wandb

class QuantizedTrainer:
    """
    A class to handle the training of a quantized model with LoRA.
    This class is designed to work with Hugging Face's Transformers library and
    the PEFT library for parameter-efficient fine-tuning.
    """
    def __init__(self, model_name: str, quant_config_path: Path, lora_config_path: Path, training_config_path: Path,
             dataset_path: Path, output_dir: Path, project_name: str = "summary_model") -> None:
        """
        Initialize the QuantizedTrainer.

        Args:
            model_name (str): The name of the model to be trained.
            quant_config_path (Path): Path to the quantization configuration file.
            lora_config_path (Path): Path to the LoRA configuration file.
            training_config_path (Path): Path to the training configuration file.
            dataset_path (Path): Path to the dataset file.
            output_dir (Path): Directory where the model will be saved.
            project_name (str): Name of the WandB project.
        """
        self.model_name = model_name
        self.quant_config = self._load_config(quant_config_path)
        self.lora_config = self._load_config(lora_config_path)
        self.training_config = self._load_config(training_config_path)
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.project_name = project_name

        wandb.login()
        wandb.init(project=self.project_name)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self.load_model()
        self.model = self.apply_lora(self.model)
    
    def _load_config(self, path: Path) -> dict:
        """
        Load the configuration file.
        This method assumes the configuration file is in JSON format.

        Args:
            path (Path): Path to the configuration file.

        Returns:
            dict: The loaded configuration.
        """
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def load_model(self) -> torch.nn.Module:
        """
        Load the model with quantization.
        This method uses the BitsAndBytesConfig to load the model in 4-bit quantization.

        Returns:
            torch.nn.Module: The loaded model.
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.quant_config.get("load_in_4bit", True),
            bnb_4bit_use_double_quant=self.quant_config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=self.quant_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, self.quant_config.get("bnb_4bit_compute_dtype", "float16")),
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        return model

    def apply_lora(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply LoRA to the model.

        Args:
            model (torch.nn.Module): The model to apply LoRA to.

        Returns:
            torch.nn.Module: The model with LoRA applied.
        """
        config = LoraConfig(
            r=self.lora_config["r"],
            lora_alpha=self.lora_config["lora_alpha"],
            target_modules=self.lora_config["target_modules"],
            lora_dropout=self.lora_config["lora_dropout"],
            bias=self.lora_config["bias"],
            task_type=TaskType[self.lora_config["task_type"]]
        )
        return get_peft_model(model, config)

    def load_dataset(self) -> Dataset:
        """
        Load the dataset from the specified path.
        This method assumes the dataset is in JSON format.
        
        Returns:
            Dataset: The loaded dataset.
        """
        with self.dataset_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
        return dataset

    def tokenize_function(self, example: dict) -> dict:
        """
        Tokenize the input text.
        This method uses the tokenizer to convert the input text into tokens.

        Args:
            example (dict): The input text to tokenize.
        
        Returns:
            dict: The tokenized input text.
        """
        return self.tokenizer(
            example["input"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )

    def prepare_data(self) -> tuple[Dataset, Dataset]:
        """
        Prepare the dataset for training.
        This method tokenizes the dataset and splits it into training and evaluation sets.

        Returns:
            tuple[Dataset, Dataset]: The training and evaluation datasets.
        """
        dataset = self.load_dataset()

        system_prompt = "<|system|>\nYou are a helpful AI specialized in summarizing code.\n"

        def merge_inputs(example):
            code_snippets = "\n".join(
                snippet["code"] for snippet in example["input_group"]
            )
            user_prompt = f"<|user|>\n{example['synthetic_prompt']}\n\nRETRIEVED FRAGMENTS:\n{code_snippets}\n"
            assistant_prompt = "<|assistant|>\n"

            example["input"] = system_prompt + user_prompt + assistant_prompt
            example["output"] = example["generated_summary"]
            return example

        dataset = dataset.map(merge_inputs)
        dataset = dataset.remove_columns(["synthetic_prompt", "input_group", "generated_summary"])

        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)

        split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
        return split_dataset["train"], split_dataset["test"]

    def train(self) -> None:
        """
        Train the model.
        This method initializes the Trainer and starts the training process.
        Results are logged to WandB.
        """
        train_dataset, eval_dataset = self.prepare_data()
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            run_name=self.project_name,
            **self.training_config
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        trainer.train()
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        wandb.finish()
