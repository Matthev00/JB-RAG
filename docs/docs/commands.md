# Commands

## Crate Enviroment

```sh
make create_environment
```

## Install Python Dependencies

Installs the required Python dependencies listed in `requirements.txt`.

```sh
make requirements
```

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

