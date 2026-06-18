import os
import hashlib
import chromadb
from chromadb.config import Settings
import streamlit as st

# ==========================================
# 1. DATABASE INITIALIZATION
# ==========================================
VECTOR_DB_PATH = "./chroma_db"

@st.cache_resource(show_spinner=False)
def get_chroma_client():
    return chromadb.PersistentClient(
        path=VECTOR_DB_PATH,
        settings=Settings(allow_reset=True)
    )

# ==========================================
# 2. VECTOR STORE POPULATION
# ==========================================
def populate_vector_store(chunks: list[dict], collection_name: str = "notebook_collection") -> bool:
    client = get_chroma_client()
    # Chroma handles the all-MiniLM-L6-v2 embeddings natively via fast ONNX runtime!
    collection = client.get_or_create_collection(name=collection_name)
    
    documents = []
    metadatas = []
    ids = []
    
    for chunk in chunks:
        doc_text = chunk.get("text", "").strip()
        
        if not doc_text:
            continue
            
        meta = chunk["metadata"]
        content_hash = hashlib.md5(doc_text.encode("utf-8")).hexdigest()
        
        safe_filename = meta["source_file"].replace(" ", "_")
        doc_id = f"{safe_filename}_p{meta['page_number']}_{content_hash[:10]}"
        
        documents.append(doc_text)
        metadatas.append(meta)
        ids.append(doc_id)
        
    if not documents:
        return True
        
    # Pass raw text; Chroma automatically embeds it using minimal RAM
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    return True

# ==========================================
# 3. SEMANTIC CONTEXT RETRIEVAL
# ==========================================
def retrieve_context(query: str, collection_name: str = "notebook_collection", top_k: int = 3) -> list[dict]:
    client = get_chroma_client()
    collection = client.get_collection(name=collection_name)
    
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    retrieved_context = []
    
    if results['documents'] and len(results['documents'][0]) > 0:
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        
        for doc, meta in zip(docs, metas):
            retrieved_context.append({
                "text": doc,
                "metadata": meta
            })
            
    return retrieved_context