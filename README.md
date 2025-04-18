# JB_RAG


## **Description**

Retrieval-Augmented Generation (RAG) system over a code repository for a question-answering task.

Users can use the system for question answering over the repository:  
- **Input**: Natural language query (question).  
- **Output**: Relevant code locations (files).  

### System 

Code is parsed into chunks using a custom CodeParser, and embeddings are generated with a SentenceTransformer model. The FAISS index enables fast similarity-based searches, supporting both radius-based and top-k retrieval methods. The system is designed to be modular, allowing easy customization of embedding models and repositories, while ensuring reproducibility through experiment tracking with Optuna and Weights & Biases.

#### LLM summaries

To enhance the user experience, the system integrates an LLM (Large Language Model) to generate natural language summaries of retrieved results. This feature is available in the web interface and is **enabled by default using Azure OpenAI** (e.g., `gpt-3.5-turbo`).

##### Configuration

You can configure the behavior in `src/config.py`:

- `USE_OPENAI = True` *(default)*  
  Uses **Azure OpenAI API** for high-quality summaries.

- `USE_OPENAI = False`  
  Switches to a **local lightweight model** (e.g., `TinyLlama`) for offline inference.

##### 🧠 Local Model Settings

When using a local model, the following config variables apply:

- `SUMMARY_MODEL` – name or path to the model (e.g., `"TinyLlama"` or `"TheBloke/Mistral-7B-Instruct-GGUF"`)
- `DEVICE` – hardware device (e.g., `"cpu"`, `"cuda"`, or `"mps"`)

The default local model is `TinyLlama`, allowing for CPU-friendly inference. However, you can replace it with any instruction-tuned LLaMA-style model (e.g., `LLaMA 2`, `Mistral`, etc.) for better summary quality.

---
   **Note** This project uses **Azure OpenAI** by default, as I have access to it through the [Azure for Students](https://azure.microsoft.com/free/students/) program, which provides free credits for cloud services.   
💡 **Note:** To use Azure OpenAI, make sure your `.env` file contains:

```env
AZURE_OPENAI_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
```


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
   - A predefined list of candidate terms is used to expand the query. The system calculates the semantic similarity between the query and these candidate terms using a SentenceTransformer model and selects the most relevant terms to expand the query.

2. **WordNet-based Expansion**:
   - This approach uses the WordNet lexical database to find synonyms and related terms for words in the query.

**Results - latency/quality trade-offs**  
The addition of Query Expansion based on static candidate terms resulted in a very slight improvement in recall, increasing it from **0.50343** to **0.50833**. However, this improvement came at a significant cost in terms of latency, which increased by approximately **3 times**.

**Conclusions**

1. **When to Use Query Expansion**:
   - Query Expansion based on static candidate terms may be beneficial in scenarios where **recall is critical**, and even a small improvement in recall can justify the additional latency

2. **When to Avoid Query Expansion**:
   - If **latency is a critical factor**, such as in real-time systems.

3. **Trade-offs**:
   - The decision to use Query Expansion should be based on the specific requirements of the application. For systems prioritizing **quality over speed**, Query Expansion can be a valuable addition. However, for systems where **speed is paramount**, it is better to avoid it.


#### Adding Reranker
I attempted to add a reranker to improve the ranking of search results. However, since the FAISS `search` method already uses `range_search` and sorts the results by distance, adding a reranker based on the same tokenizer and cosine similarity calculations did not bring any noticeable improvements.

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

---

## **How to use system**

### First, you need to prepare the knowledge base

```sh
make prepare_kb
```
This will download the repository specified in `src/config.py` as `REPO_URL`(this is the repository mentioned in the task). If you want to test with a different repository, you can simply change the `REPO_URL`.

### Run UI 
run

```sh
uv run streamlit run src/main.py --server.fileWatcherType=none
```
**Default values are, in my opinion, the best based on the reports.**  
This will launch a web-based interface where you can interact with the system. Open the provided URL in your browser (e.g., http://localhost:8501).

**How to Use the Interface**

1. **Select the Search Type**:
   - Choose between:
     - **Radius Search**: Searches for results within a specified similarity radius.
     - **Top K Search**: Retrieves the top K most similar results.

2. **Set Parameters**:
   - For **Radius Search**, adjust the radius using the slider.
   - For **Top K Search**, select the number of top results to retrieve.

3. **Query Expansion**:
   - Choose a query expansion method:
     - **None**: No query expansion.
     - **WordNet**: Expands the query using synonyms from WordNet.
     - **Candidate Terms**: Expands the query using predefined candidate terms.

4. **Enter Your Query**:
   - Type your query in the input box.

5. **Search**:
   - Click the "Search" button to retrieve results.

6. **View Results**:
   - The results will be displayed as a list of file paths. Each result corresponds to a code chunk that matches your query.

## Evaluate model

1. Run `make evaluate`.

### Reproduce experiments

You can reproduce experiments by preparing validation dataset by downloading file from [here](https://drive.google.com/file/d/1PiiordcQJwgv4MfT1vl-Omn8DeCdlAB3/view) and saving it as `/data/escrcpy_val.json`. Then you can run `make hyperparameter_experiment` and `make query_expansion_experiment`.

---


## Project Repository Organization 

```
├── LICENSE                 <- Open-source license
├── README.md               <- Project overview and instructions
├── .gitignore              <- Files/folders ignored by Git
├── .env                    <- Environment variables (not committed)
├── requirements.txt        <- Python dependencies
├── pyproject.toml          <- Project metadata and build system
├── setup.py                <- Installation script
├── uv.lock                 <- Lock file for dependencies
│
├── data/                   <- Input/output data
│   ├── embeddings/         <- Chunked and embedded documents
│   ├── faiss/              <- FAISS index files (vector DB)
│   ├── repos/              <- Cloned repositories for RAG
│   ├── .gitkeep            <- Placeholder to keep folder in git
│   └── escrcpy_val.json    <- Example data file
│
├── docs/                   <- Markdown documentation and mkdocs site
│
├── jb_rag.egg-info/        <- Package metadata for distribution
│
├── app_utils/              <- App-specific utility scripts
│   └── answer_summary.py   <- Summarizes retrieved answers
│
├── evaluation/             <- Model evaluation modules
│   ├── dataset.py
│   ├── evaluator.py
│   ├── hyperparameters_experiment.py
│   ├── query_expansion_experiment.py
│   └── utils.py
│
├── preprocessing/          <- Data parsing and embedding
│   ├── code_parser.py
│   ├── embedding_generator.py
│   └── repo_loader.py
│
├── retriever/              <- Retrieval and search logic
│   ├── candidate_terms.py
│   ├── faiss_search.py
│   ├── query_expander.py
│   └── __init__.py
│
├── config.py               <- Global configuration file
├── evaluate.py             <- Script for model evaluation
├── knowledge_base_preparation.py <- Prepares vector DB from repos
├── main.py                 <- Main entry point for running the system
│
└── wandb/                  <- Weights & Biases experiment logs

```

---

## **Author**

- Mateusz Ostaszewski

---
