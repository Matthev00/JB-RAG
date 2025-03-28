# JB_RAG


## **Description**

Retrieval-Augmented Generation (RAG) system over a code repository for a question-answering task.

Users can use the system for question answering over the repository:
- **Input**: Natural language query (question).
- **Output**: Relevant code locations (files).

### System 

Code is parsed into chunks using a custom CodeParser, and embeddings are generated with a SentenceTransformer model. The FAISS index enables fast similarity-based searches, supporting both radius-based and top-k retrieval methods. The system is designed to be modular, allowing easy customization of embedding models and repositories, while ensuring reproducibility through experiment tracking with Optuna and Weights & Biases.

The code documentation is generated using MkDocs and hosted on [GitHub Pages](https://matthev00.github.io/JB-RAG/).

### Experiments 

#### Hyperparameters tuning 

To optimize retrieval, multiple experiments were conducted by varying three key parameters:
- max_chunk_size: Size of code/document chunks.  
- top_k: Number of retrieved results.  
- radius: Similarity threshold in vector space.  

The evaluation metrics included Recall@10, Precision@10, F1@10, and MRR, with Recall@10 being the primary metric due to task requirements.

  [REPORT WITH CONCLUSIONS](https://api.wandb.ai/links/MY_EXPERIMENTS/1n77w34b)

I will continue my experiments using the model with the highest Recall@10, as this is the main optimization goal of the project. Additionally, I will include the model with the best F1@10 for comparison purposes.

#### Adding expand Query
The system implements Query Expansion to enhance the quality of search results by enriching the user's query with additional related terms. Two approaches to Query Expansion are supported:
1. **Static Candidate Terms**:
   - A predefined list of candidate terms is used to expand the query. These terms cover a wide range of topics, such as programming concepts (e.g., "function", "class"), user interface elements (e.g., "button", "dialog"), and system-related terms (e.g., "network", "configuration"). The system calculates the semantic similarity between the query and these candidate terms using a SentenceTransformer model and selects the most relevant terms to expand the query.

2. **WordNet-based Expansion**:
   - This approach uses the WordNet lexical database to find synonyms and related terms for words in the query. It dynamically generates expanded queries by including these related terms, which can improve recall for queries with synonyms or alternative phrasings.

**Results**


#### Adding Reranker



---

## **Key Functionalities**

- Ability to search through a repository after preparing the knowledge base.
- Reproducibility of experiments.
- Option to change the embedding model by modifying the `EMBEDDING_MODEL` variable in the `src/config.py` file.
- Option to change the repository to be searched by updating the `REPO_URL` variable in the `src/config.py` file. After making this change, you need to run `make prepare_kb again`.

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

## Evaluate model
1. Downloading file from [here](https://drive.google.com/file/d/1PiiordcQJwgv4MfT1vl-Omn8DeCdlAB3/view)
2. Run `make evaluation`.

### Reproduce experiments

You can reproduce experiments by preparing validation dataset by downloading file from [here](https://drive.google.com/file/d/1PiiordcQJwgv4MfT1vl-Omn8DeCdlAB3/view) and saving it as `/data/escrcpy_val.json`. Then you can run `make hyperparameter_experiment`.

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
