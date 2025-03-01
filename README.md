# PDF-Based RAG Chatbot

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** using **FAISS** for indexing, **Ollama** for embeddings & responses, and **LangChain** for PDF processing.

## Features
- Extracts and processes text from PDFs.
- Splits text into chunks for efficient retrieval.
- Uses FAISS to index and search relevant document chunks.
- Generates answers using the **Llama 3.2** model.

## Requirements
Install dependencies using:
```bash
pip install faiss-cpu ollama langchain-community langchain-ollama numpy pickle5
```

## How It Works
1. **PDF Loading**: Extracts text from PDFs.
2. **Chunking**: Splits text into overlapping 500-character chunks.
3. **Embedding**: Converts text chunks into vector embeddings.
4. **Indexing**: Stores embeddings in a FAISS index for fast retrieval.
5. **Querying**: Retrieves relevant chunks & generates AI-powered responses.

## Usage
Run the script and enter your query:
```bash
python rag.py
```

## Future Improvements
- Integrate a **Streamlit UI**.
- Deploy on **Netlify** or **Hugging Face Spaces**.

---
Developed by **Snehil Deep** 🚀

