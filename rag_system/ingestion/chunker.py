# rag_system/ingestion/chunker.py

import os
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

# Try to import nltk, fallback to simple splitting if not available
try:
    import nltk
    # Download nltk data if not available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("[Info] Downloading NLTK punkt tokenizer...")
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            print(f"[Warning] Could not download NLTK punkt: {e}. Using simple sentence splitting.")
            nltk = None
    HAS_NLTK = True
except ImportError:
    print("[Warning] NLTK not available. Using simple sentence splitting.")
    HAS_NLTK = False
    nltk = None

def split_into_sentences(text):
    """Split text into sentences using NLTK or fallback method"""
    if HAS_NLTK and nltk:
        try:
            sentences = nltk.sent_tokenize(text)
            return sentences
        except Exception as e:
            print(f"[Warning] NLTK tokenization failed: {e}. Using fallback.")
    
    # Fallback to simple sentence splitting
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    return [s.strip() + '.' for s in sentences if s.strip()]

def create_chunks_with_overlap(sentences, chunk_size, overlap):
    """Create chunks with sentence boundaries and overlap"""
    chunks = []
    current_chunk = []
    current_length = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_length = len(sentence)
        
        # If single sentence exceeds chunk size, split it
        if sentence_length > chunk_size:
            # Add current chunk if it has content
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split long sentence into smaller parts
            words = sentence.split()
            temp_chunk = []
            temp_length = 0
            for word in words:
                word_with_space = word + ' '
                if temp_length + len(word_with_space) <= chunk_size:
                    temp_chunk.append(word)
                    temp_length += len(word_with_space)
                else:
                    if temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                    temp_chunk = [word]
                    temp_length = len(word_with_space)
            if temp_chunk:
                chunks.append(' '.join(temp_chunk))
            i += 1
            continue
        
        # Check if adding this sentence would exceed chunk size
        if current_length + sentence_length + 1 > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            overlap_sentences = []
            overlap_length = 0
            # Go back to include overlap sentences
            for j in range(len(current_chunk) - 1, -1, -1):
                sent = current_chunk[j]
                if overlap_length + len(sent) + 1 <= overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_length += len(sent) + 1
                else:
                    break
            
            current_chunk = overlap_sentences
            current_length = overlap_length
        else:
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_length + 1
            i += 1
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_files(input_dir, output_dir, chunk_size=None, chunk_overlap=None):
    """Process files with semantic-aware chunking (memory-efficient for large files)"""
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = CHUNK_OVERLAP
    
    os.makedirs(output_dir, exist_ok=True)

    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    total_files = len(txt_files)
    
    if total_files == 0:
        print("  No text files found to chunk.")
        return

    processed_count = 0
    skipped_count = 0
    
    for idx, filename in enumerate(txt_files, 1):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".txt", "_chunks.txt"))

        # Check if chunks already exist and are up to date
        if os.path.exists(output_path):
            input_time = os.path.getmtime(input_path)
            output_time = os.path.getmtime(output_path)
            if output_time >= input_time:
                skipped_count += 1
                print(f"  Skipping {filename} - already chunked ({idx}/{total_files})")
                continue

        # Calculate progress percentage (25-50% for chunking)
        progress = 25 + int((idx / total_files) * 25)
        
        print(f"  Chunking {filename} ({idx}/{total_files}) - {progress}%")
        print(f"    Processing file in chunks to avoid memory issues...", end="", flush=True)

        # Memory-efficient chunking: process and write incrementally
        chunk_count = 0
        current_chunk = []
        current_length = 0
        overlap_buffer = []
        
        # Process file in smaller sections
        with open(input_path, "r", encoding="utf-8") as infile, \
             open(output_path, "w", encoding="utf-8") as outfile:
            
            # Read file in chunks to avoid loading entire file into memory
            buffer = ""
            chunk_read_size = 1024 * 1024  # 1MB at a time
            
            while True:
                chunk_text = infile.read(chunk_read_size)
                if not chunk_text:
                    # Process remaining buffer
                    if buffer:
                        sentences = split_into_sentences(buffer)
                        for sentence in sentences:
                            sentence_length = len(sentence)
                            
                            # If adding this sentence would exceed chunk size
                            if current_length + sentence_length + 1 > chunk_size and current_chunk:
                                # Write current chunk
                                chunk_text = ' '.join(current_chunk)
                                if len(chunk_text.strip()) > 50:
                                    outfile.write(chunk_text.strip() + "\n")
                                    chunk_count += 1
                                
                                # Start new chunk with overlap
                                overlap_buffer = current_chunk[-3:] if len(current_chunk) >= 3 else current_chunk
                                current_chunk = overlap_buffer.copy()
                                current_length = sum(len(s) for s in current_chunk)
                            
                            current_chunk.append(sentence)
                            current_length += sentence_length + 1
                    break
                
                buffer += chunk_text
                
                # Process complete sentences from buffer
                while True:
                    # Try to find sentence boundaries
                    period_idx = buffer.find('. ')
                    exclamation_idx = buffer.find('! ')
                    question_idx = buffer.find('? ')
                    
                    # Find the earliest sentence boundary
                    boundaries = [i for i in [period_idx, exclamation_idx, question_idx] if i != -1]
                    if not boundaries:
                        # No complete sentence found, read more
                        break
                    
                    sentence_end = min(boundaries) + 1
                    sentence = buffer[:sentence_end].strip()
                    buffer = buffer[sentence_end:].strip()
                    
                    if not sentence:
                        continue
                    
                    sentence_length = len(sentence)
                    
                    # If adding this sentence would exceed chunk size
                    if current_length + sentence_length + 1 > chunk_size and current_chunk:
                        # Write current chunk
                        chunk_text = ' '.join(current_chunk)
                        if len(chunk_text.strip()) > 50:
                            outfile.write(chunk_text.strip() + "\n")
                            chunk_count += 1
                        
                        # Start new chunk with overlap
                        overlap_buffer = current_chunk[-3:] if len(current_chunk) >= 3 else current_chunk
                        current_chunk = overlap_buffer.copy()
                        current_length = sum(len(s) for s in current_chunk)
                    
                    current_chunk.append(sentence)
                    current_length += sentence_length + 1
            
            # Write final chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.strip()) > 50:
                    outfile.write(chunk_text.strip() + "\n")
                    chunk_count += 1

        processed_count += 1
        print(f" ✓ {chunk_count} chunks created")
    
    if skipped_count > 0:
        print(f"  ({skipped_count} file(s) skipped - already chunked)")
