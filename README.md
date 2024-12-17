# RAG-kaggle-competition
RAG pipeline for large unstructured PDF file

Attached is a Colab notebook implementing RAG pipelione for large PDF file (approx 760 pages)

# Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline to accurately answer questions based on a large document (e.g., PDF) using open-source models and tools. The system processes the data, indexes it for efficient retrieval, and generates accurate answers with clean context and verifiable references (e.g., page numbers).

The key components include:

PDF to Context Processing: Splitting large PDFs into clean, queryable text chunks.
Vector Database Indexing: Embedding text using HuggingFace models and storing in a vector index.
Open-Source LLM Generation: Query processing using Falcon-7B-Instruct for Q&A generation.
Clean Outputs: Contextual responses, cleaned data, and page-level references.

# Key Features
1. Document Parsing and Chunking: Processes and cleans large PDFs into manageable text chunks.
2. VectorStore Index: Stores embeddings of text chunks for efficient retrieval.
3. Q&A Generation: Answers queries using retrieved context via the Falcon-7B-Instruct model.
4. References Extraction: Extracts and includes page numbers in responses for verifiable answers.

# Dependencies

!pip install pypdf
!pip install -q transformers einops accelerate langchain bitsandbytes
!pip install sentence_transformers
!pip install llama_index
!pip install llama-index-llms-huggingface
!pip install llama-index-llms-huggingface-api
!pip install langchain-community
!pip install llama-index-embeddings-langchain


# How It Works
1. System Initialization:
Loads an open-source LLM (tiiuae/falcon-7b-instruct) for generation.
Uses HuggingFace embeddings (sentence-transformers/all-mpnet-base-v2) for text embedding.
2. Document Processing:
Reads documents (book.pdf) from the data folder.
Splits the text into manageable chunks (1024 tokens) and cleans special characters.
3. Vector Index Creation
Embeds document chunks and stores them in a VectorStoreIndex for retrieval.
4. Query Engine
Accepts input questions from a queries.json file.
Retrieves the most relevant document chunks using vector similarity search.
Generates a contextual answer using the Falcon-7B-Instruct model.
5. Output Generation
Cleans the generated responses and extracted context.
Saves results (including references and page numbers) in the submission.csv file

# Key Learnings
Optimized RAG Pipelines: Efficiently combined document retrieval and generation using open-source tools.
Accurate References: Successfully extracted and included page-level references for verifiable answers.
Text Cleaning and Chunking: Addressed data preprocessing challenges for clean and queryable input.

# Future Improvements
Use more advanced chunking techniques (e.g., semantic splitting).
Enhance retrieval accuracy using hybrid search (keyword + vector).
Explore larger open-source LLMs for better answer quality.
