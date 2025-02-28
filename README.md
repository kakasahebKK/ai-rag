# RAG Application with LangChain, Qdrant, and Ollama

This application implements a Retrieval-Augmented Generation (RAG) system that:
1. Takes user questions as input
2. Retrieves relevant information from a Qdrant vector database
3. Uses the retrieved context to generate accurate answers with locally hosted Ollama models

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running locally
- Qdrant running locally or as a cloud service

## Installation

1. Clone this repository
2. Install the required packages:
```
pip install -r requirements.txt
```
3. Set up environment variables (optional, defaults are set in the code):
```
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_EMBED_MODEL="nomic-embed-text"
export OLLAMA_LLM_MODEL="llama3"
export QDRANT_URL="http://localhost:6333"
```

## Required Ollama Models

Ensure you have the necessary models pulled in Ollama:

1. For embeddings (you can choose a different embedding model):
```
ollama pull nomic-embed-text
```

2. For text generation (you can choose a different LLM):
```
ollama pull llama3
```

## Running the Application

1. Start Qdrant (if running locally):
```
docker run -p 6333:6333 qdrant/qdrant
```

2. Ensure Ollama is running:
```
ollama serve
```

3. Initialize the system with sample documents:
```
python main.py
```

4. (Optional) Add custom documents:
```
python sample_documents.py
```

## Usage

Once the application is running, you will be prompted to input questions. The system will:
1. Convert your question to a vector embedding using Ollama
2. Search the Qdrant database for similar content
3. Retrieve the most relevant document chunks
4. Send your question along with the retrieved context to Ollama
5. Return the generated answer

Type 'exit' to quit the application.

## Customization

- Modify the sample documents in `main.py` or `sample_documents.py`
- Adjust the chunk size and overlap in the `split_documents` function
- Change the retrieval parameters (like the number of chunks to retrieve) in the `setup_retrieval_qa_chain` function
- Update the prompt template to customize how the LLM processes the context and questions
- Change the Ollama models by setting environment variables or modifying the defaults in the code

## Extending the Application

You can extend this application by:
- Adding document loaders for various file types (PDFs, Word documents, websites)
- Implementing a web interface
- Adding document deletion or update functionality
- Incorporating multiple vector stores or different embedding models
- Adding user authentication for multi-user environments
