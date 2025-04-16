# CODE RAG documentation!

## Description

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
AZURE_OPENA
```

## Commands

The Makefile contains the central entry points for common tasks related to this project.

