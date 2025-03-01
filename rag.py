# import pdfplumber
# import faiss
# import numpy as np
# import ollama
# import pickle
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_ollama import OllamaEmbeddings

# # Step 1: Load and Extract Text from PDF
# def load_pdf(file_path):
#     with pdfplumber.open(file_path) as pdf:
#         text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
#     return text

# pdf_text = load_pdf("C:\\Users\\Snehil\\Desktop\\Final Research Paper.pdf")

# # Step 2: Split Text into Chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# text_chunks = text_splitter.split_text(pdf_text)

# # Save text chunks separately
# with open("text_chunks.pkl", "wb") as f:
#     pickle.dump(text_chunks, f)

# # Step 3: Generate Embeddings for Chunks
# embedding_model = OllamaEmbeddings(model="nomic-embed-text")
# embeddings = embedding_model.embed_documents(text_chunks)

# # Convert embeddings to a NumPy array
# embeddings_array = np.array(embeddings)

# # Step 4: Store Embeddings in FAISS
# dimension = embeddings_array.shape[1]
# faiss_index = faiss.IndexFlatL2(dimension)
# faiss_index.add(embeddings_array)

# # Save FAISS index
# faiss.write_index(faiss_index, "faiss_index.bin")

# # Step 5: Define FAISS Retriever
# class FAISSRetriever:
#     def __init__(self, index, texts):
#         self.index = index
#         self.texts = texts

#     def retrieve(self, query, top_k=3):
#         # Generate query embedding
#         query_embedding = embedding_model.embed_query(query)
#         query_embedding = np.array(query_embedding).reshape(1, -1)

#         if self.index.ntotal == 0:
#             return ["No relevant chunks found."]

#         distances, indices = self.index.search(query_embedding, top_k)
#         return [self.texts[i] for i in indices[0] if i < len(self.texts)]

# # Load FAISS index and text chunks
# faiss_index = faiss.read_index("faiss_index.bin")
# with open("text_chunks.pkl", "rb") as f:
#     text_chunks = pickle.load(f)

# retriever = FAISSRetriever(faiss_index, text_chunks)

# # Step 6: Generate answer using retrieved context
# def generate_answer(context, query):
#     prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
#     response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
#     return response["message"]["content"]

# # Step 7: Full RAG pipeline
# def rag_pipeline(query):
#     relevant_texts = retriever.retrieve(query)
#     context = "\n".join(relevant_texts)

#     if not context.strip():
#         return "Sorry, I couldn't find relevant information."

#     return generate_answer(context, query)

# # Step 8: Retrieve and Display Relevant Chunks + Answer
# query = input("Enter your query: ")
# relevant_texts = retriever.retrieve(query)

# print("\n--- RELEVANT CHUNKS ---\n")
# for i, chunk in enumerate(relevant_texts, 1):
#     print(f"Chunk {i}: {chunk}\n{'-'*50}")

# print("\n--- AI Answer ---\n")
# response = rag_pipeline(query)
# print("AI:", response)


import os
import faiss
import numpy as np
import ollama
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# pdf ko load aur extract krne ke liye function hai yeh
def load_pdfs(file_paths):  
    all_texts = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)  # pdf ko load krte h isse
        pages = loader.load()  # isko use krke extract krte h pages
        all_texts.extend([page.page_content for page in pages])
    return "\n".join(all_texts)  

# multiple pdfs ka path lena hai
pdf_files = [
    "C:\\Users\\Snehil\\Desktop\\Final Research Paper.pdf",
    "C:\\Users\\Snehil\\Desktop\\Snehil and Dhruv - Wildlife detection Research paper.pdf"
]

# pdfs ko combine kr rhe h 
combined_text = load_pdfs(pdf_files)

# isse chunks mai split krte hai , 500 tokens mai aur 100 tokens/characters mai taki continuity rahe
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# ab iss splitter se pdf ke text ko chunks m convert krenge
text_chunks = text_splitter.split_text(combined_text)

# print total number of chunks
print(f"\nTotal Chunks Created: {len(text_chunks)}\n")

# later use ke liye ek file mai save krenge
with open("text_chunks.pkl", "wb") as f:
    pickle.dump(text_chunks, f)

# embedding krenge ab har chunk ka    
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
embeddings = embedding_model.embed_documents(text_chunks)    

# faiss processing ke liye numpy array m convert krna hota h 
embeddings_array = np.array(embeddings)

# faiss ka indexing krna h phir embedding krni h
dimension = embeddings_array.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings_array)

# later retrieval ke liye faiss index ko ek file m save kr rhe h
faiss.write_index(faiss_index, "faiss_index.bin")

# ab query ke liye most important text chunks ko retrieve krne ke liye class banayenge
class FAISSRetriever:
    
    def __init__(self, index, texts):
        self.index = index
        self.texts = texts
        
    # query mai se top k chunks retrieve krta h 
    def retrieve(self, query, top_k=3):
        query_embedding = embedding_model.embed_query(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)  # reshaping krte h taki faiss ka input format match krske
        
        if self.index.ntotal == 0:
            return ["No relevant chunks found."]
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        return [self.texts[i] for i in indices[0] if i < len(self.texts)]
    
# ab faiss index aur text chunks ko load krnege query retrieval ke liye
faiss_index = faiss.read_index("faiss_index.bin")
with open("text_chunks.pkl", "rb") as f:
    text_chunks = pickle.load(f)  # load text chunks from saved file
    
retriever = FAISSRetriever(faiss_index, text_chunks)    

def generate_answer(context, query):
    
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    
    return response["message"]["content"]   
        
        
def rag_pipeline(query):
    
    relevant_texts = retriever.retrieve(query)
    context = "\n".join(relevant_texts)
    
    if not context.strip():
        return generate_answer(context, query) 
    
    return generate_answer(context, query)

# user ka query lekar response denge ab
query = input("\nEnter your query: ")
relevant_texts = retriever.retrieve(query)

print("\n---------- RELEVANT CHUNKS ----------\n")
print(f"Total Chunks Retrieved: {len(relevant_texts)}\n")
for i, chunk in enumerate(relevant_texts, 1):
    print(f"Chunk {i}:\n{chunk}\n{'_'*50}")
    
print("\n--- AI Answer ---\n")
response = rag_pipeline(query)
print("AI:", response)