# sample_documents.py
# This script demonstrates how to load custom documents into the RAG system

import os
from main import initialize_qdrant, initialize_embeddings, create_documents, split_documents, create_vector_store

# You can add your own custom documents here
custom_documents = [
    "The Qdrant vector database is designed for storing, searching, and managing vectors with an optional payload. "
    "It is optimized for extended filtering support and horizontal scalability. "
    "Qdrant can be used as a vector similarity search engine for recommendation systems, semantic document search, "
    "and other applications requiring efficient vector similarity operations.",
    
    "LangChain is a framework for developing applications powered by language models. "
    "It enables applications that are context-aware, reason, and interface with external tools. "
    "LangChain provides modules for working with language models, document loaders, text splitters, embeddings, and vector stores.",
    
    "Retrieval-Augmented Generation (RAG) combines retrieval mechanisms with text generation models. "
    "In RAG systems, a retrieval component fetches relevant information from a knowledge base or external source, "
    "which is then incorporated as context for a language model to generate more informed, accurate, and up-to-date responses.",
    
    "Ollama is an open-source project that allows users to run large language models locally. "
    "It provides an easy-to-use interface for downloading, running, and managing various language models "
    "on personal computers. Ollama supports a variety of models like Llama, Mistral, and others.",
    
    "Vector embeddings are numerical representations of data (like text, images, or audio) in a high-dimensional space. "
    "In these spaces, semantically similar items are positioned closer together. "
    "For text data, embeddings capture meaning, allowing operations like finding similar documents "
    "or understanding relationships between concepts.",
]

def load_custom_documents():
    """Load custom documents into the RAG system"""
    print("Loading custom documents into the RAG system...")
    
    # Initialize Qdrant client
    client = initialize_qdrant()
    
    # Initialize embeddings
    embeddings = initialize_embeddings()
    
    # Process documents
    documents = create_documents(custom_documents)
    split_docs = split_documents(documents)
    
    # Create vector store with documents
    create_vector_store(client, embeddings, split_docs)
    
    print("Custom documents loaded successfully!")

if __name__ == "__main__":
    load_custom_documents()
