# main.py

import os
from typing import List, Dict, Any

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore as Qdrant
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaLLM as Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

from qdrant_client import http as qdrant_http
from qdrant_client import QdrantClient


# Environment variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DOCUMENTS_DIR = "./documents"
COLLECTION_NAME = "documents3"

def initialize_qdrant() -> QdrantClient:
    """Initialize Qdrant client"""
    return QdrantClient(url=QDRANT_URL)

def load_sample_documents() -> List[str]:
    """Load all markdown documents from DOCUMENTS_DIR"""
    documents = []
    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith('.md'):
            file_path = os.path.join(DOCUMENTS_DIR, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())
    return documents

def create_documents(texts: List[str]) -> List[Dict[str, Any]]:
    """Convert texts to document format"""
    return [{"page_content": text, "metadata": {"source": f"document_{i}"}} for i, text in enumerate(texts)]

def split_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    texts = [doc["page_content"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    
    split_texts = text_splitter.create_documents(texts, metadatas=metadatas)
    return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in split_texts]

def initialize_embeddings():
    """Initialize Ollama embeddings"""
    return OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_EMBED_MODEL
    )

def create_vector_store(client, embeddings, documents):
    """Create or update vector store with documents"""
    # Get the embedding dimension by making a test embedding
    test_embedding = embeddings.embed_query("test")
    embedding_dimension = len(test_embedding)
    
    try:
        # Check if collection exists
        client.get_collection(COLLECTION_NAME)
        print(f"Collection {COLLECTION_NAME} exists. Adding documents.")
    except Exception:
        # Create collection if it doesn't exist
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qdrant_http.models.VectorParams(
                size=embedding_dimension,
                distance=qdrant_http.models.Distance.COSINE
            )
        )
        print(f"Created collection {COLLECTION_NAME} with dimension {embedding_dimension}")
    
    # Convert to format expected by Qdrant
    texts = [doc["page_content"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    
    # Create Qdrant vector store
    Qdrant.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
    )
    print(f"Added {len(texts)} document chunks to Qdrant")

def setup_retrieval_qa_chain(embeddings):
    """Set up retrieval QA chain with custom prompt"""
    # Connect to existing Qdrant collection
    vectorstore = Qdrant(
        client=QdrantClient(url=QDRANT_URL),
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Custom prompt template
    prompt_template = """
    You are a helpful AI assistant. Use the following context to answer the user's question. 
    If the answer is not in the context, just say you don't know and don't make up information.
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create chain with Ollama LLM
    llm = Ollama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_LLM_MODEL,
        temperature=0
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

def initialize_system():
    """Initialize the entire system"""
    print("Initializing RAG system...")
    
    # Initialize Qdrant client
    client = initialize_qdrant()
    
    # Load sample documents
    texts = load_sample_documents()
    documents = create_documents(texts)
    split_docs = split_documents(documents)
    
    # Initialize embeddings
    embeddings = initialize_embeddings()
    
    # Create vector store with documents
    create_vector_store(client, embeddings, split_docs)
    
    # Setup retrieval QA chain
    qa_chain = setup_retrieval_qa_chain(embeddings)
    
    print("RAG system initialized successfully!")
    return qa_chain

def query_system(qa_chain, question: str) -> str:
    """Query the RAG system with a question"""
    result = qa_chain.invoke({"query": question})
    return result["result"]

def main():
    """Main function to run the RAG system"""
    # Initialize the system
    qa_chain = initialize_system()
    
    # Interactive loop
    print("\nRAG System is ready. Type 'exit' to quit.")
    while True:
        question = input("\nEnter your question: ")
        if question.lower() == 'exit':
            break
        
        answer = query_system(qa_chain, question)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()