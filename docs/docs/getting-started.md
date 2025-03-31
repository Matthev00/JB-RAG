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
This will also download the model defined in src/config.py (e.g. tiiuae/falcon-7b-instruct). This might take some time depending on your internet speed. You only need this if you want to use full LLM features like query explanations. If you're only interested in path-based retrieval or running evaluations — you can skip this step and just run `make requirements`.


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
