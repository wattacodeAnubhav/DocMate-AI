import os
import re
import fitz  # PyMuPDF
import gc    # Garbage Collection for memory management

# ==========================================
# 1. TEXT PROCESSING & CHUNKING
# ==========================================

SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')

# This function takes the raw text of a page and splits it into semantically coherent chunks based on sentence boundaries.
def semantic_chunk_page_text(text: str, metadata: dict, max_chunk_size: int = 1000, overlap_sentences: int = 1) -> list[dict]:
    """
    Splits text into chunks based on natural semantic boundaries (sentences).
    Groups sentences until the max_chunk_size is reached, ensuring context is never cut in half.
    """
    # Use the pre-compiled pattern instead of re.split()
    raw_sentences = SENTENCE_SPLIT_PATTERN.split(text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_length = len(sentence)
        
        # If adding the next sentence exceeds the limit, commit the current chunk
        if current_length + sentence_length > max_chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source_file": metadata["source_file"],
                    "page_number": metadata["page_number"]
                }
            })
            
            
            num_sentences = len(current_chunk)
            
            actual_overlap = min(overlap_sentences, num_sentences - 1)
            
            i = i - actual_overlap
            
            current_chunk = []
            current_length = 0
            continue  
            
        current_chunk.append(sentence)
        current_length += sentence_length + 1  
        i += 1
        
    # Commit any remaining sentences in the final chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "source_file": metadata["source_file"],
                "page_number": metadata["page_number"]
            }
        })
        
    return chunks

# ==========================================
# 2. DOCUMENT INGESTION PIPELINE
# ==========================================
# Batch size reduced to 5 to safely run on 8GB RAM without using disk swap
def process_document_pipeline(file_path: str, max_chunk_size: int = 1000, overlap_sentences: int = 1, batch_size: int = 5, progress_callback=None):
    """
    An enterprise-grade generator pipeline utilizing semantic chunking.
    Reads massive PDFs in memory-safe batches and yields context-aware chunks.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file path {file_path} does not exist.")
        
    file_name = os.path.basename(file_path)
    
    # Open the document using PyMuPDF
    with fitz.open(file_path) as doc:
        total_pages = len(doc)
        
        # Loop through the document in memory-safe batches
        for i in range(0, total_pages, batch_size):
            batch_chunks = []
            
            # Extract and chunk ONLY the current batch of pages
            for page_num in range(i, min(i + batch_size, total_pages)):
                page = doc.load_page(page_num)
                
                raw_text = page.get_text("text")
                page_text = str(raw_text).strip() if raw_text else ""
                
                # Skip completely empty pages
                if not page_text:
                    continue
                    
                metadata = {
                    "source_file": file_name,
                    "page_number": page_num + 1
                }
                
                # Apply Semantic Chunking
                page_chunks = semantic_chunk_page_text(
                    text=page_text, 
                    metadata=metadata, 
                    max_chunk_size=max_chunk_size, 
                    overlap_sentences=overlap_sentences
                )
                batch_chunks.extend(page_chunks)
            
            # Yield the batch to be embedded
            if batch_chunks:
                yield batch_chunks
            
            # Fire the progress callback to update the Streamlit UI
            if progress_callback:
                progress_callback(i + batch_size)
                
            # FORCE clear memory to prevent RAM usage from stacking
            del batch_chunks
            gc.collect()