# Getting Started

## Prerequisites

Before starting with the project, make sure you have installed all the required dependencies. You can do this by running the following command:

```sh
make create_environment
```

**Option A – Basic setup (no LLM)**
```sh
make requirements
```

**Option B – Full setup with LLM**
```sh
make requirements-llm
```
By default, the system uses the **OpenAI API** to generate natural language summaries of retrieved code files.

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


## Step 1: Prepare Knowladge Base

```sh
make prepare_kb
```

## Step 2: Inference with System

run

```sh
uv run src/main.py
```

Now you can input your question, and the system will return an answer with paths to the relevant files.

## Step 3:(Optional) Replicate Experiment
You can reproduce experiments by preparing validation dataset by downloading file from [here](https://drive.google.com/file/d/1PiiordcQJwgv4MfT1vl-Omn8DeCdlAB3/view) and saving it as `/data/escrcpy_val.json`. Then you can run `make hyperparameter_experiment` and `make query_expansion_experiment` or even `make reranker_experiment`.
**Important** - before running experiment you need to do step 1 Prepare Knowladge Base
