# rag_system/ingestion/chunker.py

import os
import re
from tqdm import tqdm
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

def normalize_text(text):
    """Normalize text by joining broken lines and fixing spacing"""
    import re
    # Fix merged words (lowercase followed by capital - like "beenimportant" -> "been important")
    text = re.sub(r'([a-z])([A-Z][a-z]+)', r'\1 \2', text)
    
    # Fix spaces within words (common PDF extraction issue)
    # Pattern: short word sequences with spaces (likely broken words)
    # Only fix very short sequences to avoid breaking legitimate spaces
    text = re.sub(r'\b([a-zA-Z]{1,2})\s+([a-zA-Z]{1,2})\b', r'\1\2', text)
    
    # Replace multiple newlines with single space
    text = text.replace('\n', ' ')
    # Fix multiple spaces
    text = ' '.join(text.split())
    return text.strip()

def detect_structured_markers(text):
    """Detect structured content markers like chapters, sections"""
    markers = []
    # Common patterns for structured content
    patterns = [
        (r'Chapter\s+\d+', 'chapter'),
        (r'CHAPTER\s+\d+', 'chapter'),
        (r'Section\s+\d+', 'section'),
        (r'SECTION\s+\d+', 'section'),
        (r'Part\s+\d+', 'part'),
        (r'Appendix\s+[A-Z]', 'appendix'),
    ]
    
    for pattern, marker_type in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            markers.append((match.start(), match.end(), marker_type, match.group()))
    
    return sorted(markers, key=lambda x: x[0])

def preserve_structured_context(text, chunk_size, overlap):
    """Ensure structured markers (chapters, sections) are preserved in chunks"""
    markers = detect_structured_markers(text)
    if not markers:
        return None  # No structured content, use normal chunking
    
    # Split text preserving markers
    chunks = []
    current_pos = 0
    
    for marker_start, marker_end, marker_type, marker_text in markers:
        # Add text before marker if significant
        if marker_start > current_pos:
            pre_text = text[current_pos:marker_start].strip()
            if len(pre_text) > 50:
                chunks.append(pre_text)
        
        # Include marker and following text
        # Find next marker or end of text
        next_marker_start = len(text)
        marker_idx = markers.index((marker_start, marker_end, marker_type, marker_text))
        if marker_idx + 1 < len(markers):
            next_marker_start = markers[marker_idx + 1][0]
        
        marker_chunk = text[marker_start:next_marker_start].strip()
        if marker_chunk:
            chunks.append(marker_chunk)
        
        current_pos = next_marker_start
    
    # Add remaining text
    if current_pos < len(text):
        remaining = text[current_pos:].strip()
        if remaining:
            chunks.append(remaining)
    
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
        
        # Get file size for progress tracking
        file_size = os.path.getsize(input_path)
        file_size_mb = file_size / (1024 * 1024)
        
        # Read entire file and normalize (for better sentence detection)
        # But process in sections to avoid memory issues
        chunk_count = 0
        current_chunk = []
        current_length = 0
        bytes_processed = 0
        
        with open(input_path, "r", encoding="utf-8") as infile, \
             open(output_path, "w", encoding="utf-8") as outfile:
            
            # Create progress bar
            pbar = tqdm(
                total=file_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=f"    Chunking",
                bar_format='{l_bar}{bar}| {percentage:3.0f}% [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
                leave=False,
                miniters=1024*50  # Update every 50KB to reduce overhead
            )
            
            # Read file in sections, normalize each section
            section_size = 500 * 1024  # 500KB sections
            text_buffer = ""
            
            while True:
                section = infile.read(section_size)
                if not section and not text_buffer:
                    break
                
                # Update progress bar with bytes read (approximate)
                bytes_read = len(section.encode('utf-8'))
                bytes_processed += bytes_read
                pbar.update(bytes_read)
                # Update postfix to show current chunk count
                if chunk_count > 0:
                    pbar.set_postfix_str(f'chunks: {chunk_count}')
                
                # Add section to buffer
                text_buffer += section
                
                # Process complete sentences from buffer
                # Look for sentence endings: . ! ? followed by space and capital letter or end of text
                while True:
                    # Find sentence boundaries more intelligently
                    # Look for: . ! ? followed by space and capital, or end of text
                    best_match = None
                    best_pos = len(text_buffer)
                    
                    # Try to find sentence endings
                    for end_char in ['.', '!', '?']:
                        pos = text_buffer.find(end_char, 0, best_pos)
                        while pos != -1:
                            # Check if followed by space and capital letter, or end of text
                            if pos + 1 < len(text_buffer):
                                next_char = text_buffer[pos + 1]
                                if next_char in [' ', '\n', '\t']:
                                    # Check next non-whitespace character
                                    next_non_ws = pos + 2
                                    while next_non_ws < len(text_buffer) and text_buffer[next_non_ws] in [' ', '\n', '\t']:
                                        next_non_ws += 1
                                    if next_non_ws >= len(text_buffer):
                                        # End of buffer
                                        best_match = pos + 1
                                        best_pos = pos
                                        break
                                    elif text_buffer[next_non_ws].isupper() or text_buffer[next_non_ws].isdigit():
                                        # Capital letter or number - likely sentence boundary
                                        best_match = pos + 1
                                        best_pos = pos
                                        break
                            else:
                                # End of buffer
                                best_match = pos + 1
                                best_pos = pos
                                break
                            
                            # Continue searching
                            pos = text_buffer.find(end_char, pos + 1, best_pos)
                    
                    # If no sentence boundary found, read more
                    if best_match is None:
                        # If we've read the whole file, process remaining buffer
                        if not section:
                            best_match = len(text_buffer)
                        else:
                            break
                    
                    # Extract sentence
                    sentence_text = text_buffer[:best_match].strip()
                    text_buffer = text_buffer[best_match:].strip()
                    
                    # Normalize the sentence
                    sentence_text = normalize_text(sentence_text)
                    
                    if not sentence_text or len(sentence_text) < 10:
                        continue
                    
                    sentence_length = len(sentence_text)
                    
                    # If adding this sentence would exceed chunk size
                    if current_length + sentence_length + 1 > chunk_size and current_chunk:
                        # Write current chunk
                        chunk_text = ' '.join(current_chunk)
                        chunk_text = normalize_text(chunk_text)
                        if len(chunk_text.strip()) >= 100:  # Minimum meaningful chunk
                            outfile.write(chunk_text.strip() + "\n\n")
                            chunk_count += 1
                            # Update progress bar with chunk count
                            pbar.set_postfix_str(f'chunks: {chunk_count}')
                        
                        # Start new chunk with proper overlap
                        # Calculate overlap based on chunk_overlap size, not fixed number of sentences
                        overlap_text = ""
                        overlap_length = 0
                        for sent in reversed(current_chunk):
                            test_overlap = sent + " " + overlap_text
                            if len(test_overlap.strip()) <= chunk_overlap:
                                overlap_text = test_overlap
                                overlap_length = len(overlap_text)
                            else:
                                break
                        
                        if overlap_text:
                            current_chunk = [overlap_text.strip()]
                            current_length = overlap_length
                        else:
                            current_chunk = []
                            current_length = 0
                    
                    # Add sentence to current chunk
                    current_chunk.append(sentence_text)
                    current_length += sentence_length + 1
            
            # Write final chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk_text = normalize_text(chunk_text)
                if len(chunk_text.strip()) >= 100:
                    outfile.write(chunk_text.strip() + "\n\n")
                    chunk_count += 1
            
            # Close progress bar
            pbar.close()

        processed_count += 1
        print(f"    ✓ {chunk_count} chunks created ({file_size_mb:.1f} MB processed)")
    
    if skipped_count > 0:
        print(f"  ({skipped_count} file(s) skipped - already chunked)")
