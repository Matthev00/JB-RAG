# Commands

## Crate Enviroment

```sh
make create_environment
```

## Install Python Dependencies

```sh
make requirements
```

#### ▶️ Using Azure OpenAI API

To enable Azure OpenAI-based summaries:

1. Create a `.env` file in the project root directory.
2. Add your Azure OpenAI credentials:
   ```env
   AZURE_OPENAI_KEY=your-api-key-here
   AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
3. Make sure you have billing enabled

#### ▶️ Using a Local Model
If you prefer to use a local LLM instead of the OpenAI API:
1. open `config.py`
2. set `USE_OPENAI = False`
3. Optionally configure `SUMMARY_MODEL` and `DEVICE`
**Note: On first use, the model will be downloaded from Hugging Face. This may take a few minutes depending on your internet speed**


## Delete all compiled Python files
Deletes all compiled Python files (*.pyc, *.pyo) and __pycache__ directories.
```sh
make clean
```

## Lint using flake8 and black
Lints the source code using flake8 and black. Use make format to format the code.
```sh
make lint
```

## Format source code with black
Formats the source code using black.
```sh
make format
```

## Prepare Knowladge Base

```sh
make prepare_kb
```

## Have fun with CODE RAG
```sh
streamlit run src/main.py --server.fileWatcherType=none
```

## Replicate Experiment
Evaluate on validation data
1. Download file from [here](https://drive.google.com/file/d/1PiiordcQJwgv4MfT1vl-Omn8DeCdlAB3/view)
2. Run `make evaluate`.


## Replicate Experiment
Replicates my experiments   
For this you need an acount on [Weights & Biases](https://wandb.ai/site/)
```sh
make hyperparameter_experiment
```
```sh
make query_expansion_experiment
```

## Serve documentation
Serves the documentation locally using mkdocs.
```sh
make docs_serve
```

## Build documentation
Builds the documentation using mkdocs.
```sh
make docs_build
```

