#!/usr/bin/env python3
"""
Standalone script to clean corrupted text from the graph RAG database.
"""
import re

from graph_rag.config import get_driver, close_driver, NEO4J_DB

def _sanitize_text(text: str) -> str:
    """Sanitize text for database storage."""
    if not text:
        return ""
    
    # Remove control characters (except \t, \n, \r)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', ' ', text)
    
    # Remove problematic Unicode characters
    text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)  # Zero-width characters
    text = re.sub(r'[\u2028\u2029]', '\n', text)       # Line/paragraph separators
    text = re.sub(r'[\u00a0]', ' ', text)              # Non-breaking space
    
    # Remove excessive repeated characters
    text = re.sub(r'(.)\1{10,}', r'\1', text)
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    # Ensure proper encoding
    try:
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception:
        pass
    
    return text.strip()

def _is_corrupted_text(text: str) -> bool:
    """Check if text appears to be corrupted."""
    if not text or len(text) < 10:
        return False
    
    # Check for excessive repeated characters
    if re.search(r'(.)\1{20,}', text):
        return True
    
    # Check for high ratio of non-printable characters
    printable_chars = sum(1 for c in text if c.isprintable() or c.isspace())
    if len(text) > 0 and printable_chars / len(text) < 0.7:
        return True
    
    return False

def clean_corrupted_data(dry_run=False):
    """Clean corrupted text from the database."""
    
    drv = get_driver()
    try:
        with drv.session(database=NEO4J_DB) as s:
            # Find chunks with potentially corrupted text
            chunks = s.run("""
                MATCH (c:Chunk)
                WHERE c.text IS NOT NULL AND c.text <> ""
                RETURN c.chunk_id AS chunk_id, c.text AS text, c.doc_id AS doc_id, c.heading AS heading
            """).data()
            
            corrupted_count = 0
            cleaned_count = 0
            deleted_count = 0
            
            print(f"Checking {len(chunks)} chunks for corruption...")
            
            for chunk in chunks:
                text = chunk["text"]
                heading = chunk.get("heading", "")
                
                # Check both text and heading for corruption
                text_corrupted = _is_corrupted_text(text)
                heading_corrupted = _is_corrupted_text(heading)
                
                if text_corrupted or heading_corrupted:
                    corrupted_count += 1
                    if dry_run:
                        print(f"[DRY RUN] Would clean chunk {chunk['chunk_id'][:8]}... from doc {chunk['doc_id'][:8]}...")
                        if text_corrupted:
                            print(f"  Corrupted text: {text[:100]}...")
                        if heading_corrupted:
                            print(f"  Corrupted heading: {heading[:50]}...")
                    else:
                        # Clean the text
                        cleaned_text = _sanitize_text(text)
                        cleaned_heading = _sanitize_text(heading)
                        
                        if cleaned_text and not _is_corrupted_text(cleaned_text):
                            s.run("""
                                MATCH (c:Chunk {chunk_id: $chunk_id})
                                SET c.text = $cleaned_text, c.heading = $cleaned_heading
                            """, 
                            chunk_id=chunk["chunk_id"], 
                            cleaned_text=cleaned_text,
                            cleaned_heading=cleaned_heading)
                            cleaned_count += 1
                            print(f"[CLEANED] Chunk {chunk['chunk_id'][:8]}... from doc {chunk['doc_id'][:8]}...")
                        else:
                            # If text is too corrupted, delete the chunk
                            s.run("""
                                MATCH (c:Chunk {chunk_id: $chunk_id})
                                DETACH DELETE c
                            """, chunk_id=chunk["chunk_id"])
                            deleted_count += 1
                            print(f"[DELETED] Corrupted chunk {chunk['chunk_id'][:8]}... from doc {chunk['doc_id'][:8]}...")
            
            # Also check documents
            docs = s.run("""
                MATCH (d:Document)
                WHERE d.text IS NOT NULL AND d.text <> ""
                RETURN d.doc_id AS doc_id, d.text AS text, d.title AS title
            """).data()
            
            doc_corrupted_count = 0
            doc_cleaned_count = 0
            doc_deleted_count = 0
            
            print(f"Checking {len(docs)} documents for corruption...")
            
            for doc in docs:
                text = doc["text"]
                title = doc.get("title", "")
                
                text_corrupted = _is_corrupted_text(text)
                title_corrupted = _is_corrupted_text(title)
                
                if text_corrupted or title_corrupted:
                    doc_corrupted_count += 1
                    if dry_run:
                        print(f"[DRY RUN] Would clean document {doc['doc_id'][:8]}...")
                        if text_corrupted:
                            print(f"  Corrupted text: {text[:100]}...")
                        if title_corrupted:
                            print(f"  Corrupted title: {title[:50]}...")
                    else:
                        cleaned_text = _sanitize_text(text)
                        cleaned_title = _sanitize_text(title)
                        
                        if cleaned_text and not _is_corrupted_text(cleaned_text):
                            s.run("""
                                MATCH (d:Document {doc_id: $doc_id})
                                SET d.text = $cleaned_text, d.title = $cleaned_title
                            """, 
                            doc_id=doc["doc_id"], 
                            cleaned_text=cleaned_text,
                            cleaned_title=cleaned_title)
                            doc_cleaned_count += 1
                            print(f"[CLEANED] Document {doc['doc_id'][:8]}...")
                        else:
                            s.run("""
                                MATCH (d:Document {doc_id: $doc_id})
                                DETACH DELETE d
                            """, doc_id=doc["doc_id"])
                            doc_deleted_count += 1
                            print(f"[DELETED] Corrupted document {doc['doc_id'][:8]}...")
            
            if dry_run:
                print(f"[DRY RUN] Found {corrupted_count} corrupted chunks and {doc_corrupted_count} corrupted documents")
            else:
                print("[CLEAN] Results:")
                print(f"  Chunks: {corrupted_count} corrupted, {cleaned_count} cleaned, {deleted_count} deleted")
                print(f"  Documents: {doc_corrupted_count} corrupted, {doc_cleaned_count} cleaned, {doc_deleted_count} deleted")
    finally:
        close_driver()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean corrupted text from graph RAG database")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be cleaned without making changes")
    args = parser.parse_args()
    
    clean_corrupted_data(dry_run=args.dry_run)