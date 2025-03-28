#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = JB-RAG
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	uv sync
	uv run python -m nltk.downloader wordnet
#	$(PYTHON_INTERPRETER) -m pip install -U pip
#	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Type-check using mypy
.PHONY: type-check
type-check:
	mypy src

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 src
	black --check --config pyproject.toml src

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml src


## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	$(PYTHON_INTERPRETER) -m pip install uv
	uv venv


## Prepare Knowledge Base
.PHONY: prepare_kb
prepare_kb:
	uv run src/knowledge_base_preparation.py


## Run Experiments
.PHONY: hyperparameter_experiment
hyperparameter_experiment:
	uv run src/evaluation/hyperparameters_experiment.py

.PHONY: query_expansion_experiment
query_expansion_experiment:
	uv run src/evaluation/query_expansion_experiment.py

## Serve documentation
.PHONY: docs_serve
docs_serve:
	cd docs && mkdocs serve

## Build documentation
.PHONY: docs_build
docs_build:
	cd docs && mkdocs build

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
