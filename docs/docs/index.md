# CODE RAG documentation!

## Description

Retrieval-Augmented Generation (RAG) system over a code repository for a question-answering task.

Users can use the system for question answering over the repository:  
- **Input**: Natural language query (question).  
- **Output**: Relevant code locations (files).  

### System 

Code is parsed into chunks using a custom CodeParser, and embeddings are generated with a SentenceTransformer model. The FAISS index enables fast similarity-based searches, supporting both radius-based and top-k retrieval methods. The system is designed to be modular, allowing easy customization of embedding models and repositories, while ensuring reproducibility through experiment tracking with Optuna and Weights & Biases.

#### LLM summaries
To further enhance user experience, I integrated a local LLM (Large Language Model) to generate natural language summaries of the retrieved results. This feature is available in the web interface and can be optionally enabled using a checkbox.

The specific LLM model and device configuration can be easily adjusted in the `src/config.py` file via the `SUMMARY_MODEL` and `DEVICE` variables.

By default, the system uses a very lightweight local model (`TinyLlama`) due to hardware limitations. While this allows local inference even on CPUs, the summary quality may vary. You are encouraged to replace it with another LLaMA-style instruction-tuned model (e.g., `LLaMA 2`, `Mistral`, etc.) if better performance is desired.

Alternatively, you can switch to using the **OpenAI API** by enabling `USE_OPENAI = True` in the configuration. This allows the system to generate high-quality summaries using models like `gpt-3.5-turbo` or `gpt-4`, depending on your API access.

## Commands

The Makefile contains the central entry points for common tasks related to this project.

