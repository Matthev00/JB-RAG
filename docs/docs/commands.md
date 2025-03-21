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

## Replicate Experiment
Replicates my experiments
```sh
make run_experiment
```
For this you need an acount on [Weights & Biases](https://wandb.ai/site/)

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
