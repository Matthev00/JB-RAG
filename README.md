# JB_RAG



## **Description**

Retrieval-Augmented Generation (RAG) system over a code repository for a question-answering task.

Users can use the system for question answering over the repository:
- **Input**: Natural language query (question).
- **Output**: Relevant code locations (files).

To achieve these results, several experiments with different system configurations were conducted. Below is a report of those experiments.

### Experiments results
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
---

## **Key Functionalities**

- Ability to search through a repository after preparing the knowledge base.
- Reproducibility of experiments.
- Option to change the embedding model by modifying the `EMBEDDING_MODEL` variable in the `src/config.py` file.
- ption to change the repository to be searched by updating the `REPO_URL` variable in the `src/config.py` file. After making this change, you need to run `make prepare_kb again`.

---

## **Instalation**

### Clone Repository

`git clone https://github.com/Matthev00/JB-RAG.git` 

### Prepare ENV
Before starting with the project, make sure you have installed all the required dependencies. You can do this by running the following command:

```sh
make create_environment
```

```sh
make requirements
```

---

## **How to use system**

### First, you need to prepare the knowledge base

```sh
make prepare_kb
```
This will download the repository specified in `src/config.py` as `REPO_URL`(this is the repository mentioned in the task). If you want to test with a different repository, you can simply change the `REPO_URL`.

### Have fun 
run

```sh
uv run src/main.py
```

Now you can input your question, and the system will return an answer with paths to the relevant files.

---

## Project Repository Organization 

```
├── LICENSE                   <- Open-source license
├── Makefile                  <- Makefile with project commands
├── README.md                 <- README file
├── .gitignore                <- Files and folders ignored by git
├── data                      <- Data used in the project
│   ├── embeddings            <- Chunked and embedded data
│   ├── faiss                 <- Vector db of data
│   └── repos                 <- Downloaded repos to search
│
├── docs                      <- Project documentation (Markdown files)
│   ├── api                   
│   │   ├── evaluation.md
│   │   ├── preprocessing.md
│   │   ├── retriever.md
│   │   └── scripts.md
│   ├── commands.md
│   ├── getting-started.md
│   ├── index.md
│   └── site                      <- Static site files generated by mkdocs
│
│
├── requirements.txt          <- Environment dependencies
├── setup.py                  <- Project installation script
├── pyproject.toml            <- Project configuration file
│
├── src                       <- Source code of the project (research part)
│   ├── evaluation            <- Evaluation modules
│   │   ├── dataset.py
│   │   ├── evaluator.py
│   │   └── experiments.py
│   ├── preprocessing         <- Data preprocessing modules
│   └── retriever             <- Retriever modules
│       ├── faiss_search.py
│       ├── knowledge_base_preparation.py
│       ├── config.py
│       └── main.py
│
└── wandb                     <- Directory for experiment tracking with Weights & Biases
```

---

## **Author**

- Mateusz Ostaszewski

---
