import os
import re
from pypdf import PdfReader
from pathlib import Path

def fix_pdf_text_spacing(text):
    """Fix common PDF extraction issues: spaces within words, broken lines, merged words"""
    if not text:
        return text
    
    # Fix broken words at line boundaries first (word-space-newline-word)
    text = re.sub(r'(\w)\s+\n\s*(\w)', r'\1\2', text)
    
    # Fix common merged words (words that should have spaces but don't)
    # Pattern 1: lowercase letter followed by capital letter (likely merged words)
    # Examples: "beenImportant" -> "been Important"
    text = re.sub(r'([a-z])([A-Z][a-z]+)', r'\1 \2', text)
    
    # Pattern 2: Fix common word boundaries where words got merged (all lowercase)
    # Common word endings that are often followed by another word
    # Examples: "beenimportant" -> "been important", "includingartificial" -> "including artificial"
    common_word_endings = [
        r'been', r'including', r'of', r'in', r'on', r'at', r'to', r'for', 
        r'with', r'from', r'the', r'and', r'or', r'have', r'has', r'had',
        r'that', r'this', r'these', r'those', r'which', r'what', r'where'
    ]
    for ending in common_word_endings:
        # Look for word ending followed by a word (4+ chars) - likely merged
        # Only if total length > 10 to avoid false positives
        pattern = r'\b(' + ending + r')([a-z]{4,})\b'
        def replace_func(m):
            combined = m.group(0)
            if len(combined) > 10:  # Long enough to likely be two words
                return m.group(1) + ' ' + m.group(2)
            return combined
        text = re.sub(pattern, replace_func, text, flags=re.IGNORECASE)
    
    # Fix spaces within short word sequences (common PDF issue)
    # Pattern: 1-2 letter sequence, space, 1-2 letter sequence (likely a broken word)
    text = re.sub(r'\b([a-zA-Z]{1,2})\s+([a-zA-Z]{1,2})\b', r'\1\2', text)
    
    # Fix words with spaces in the middle (like "exerci ses" -> "exercises")
    # Pattern: word part, space, word part (if both parts are 3+ chars, likely one word)
    text = re.sub(r'\b([a-zA-Z]{3,})\s+([a-zA-Z]{3,})\b', lambda m: m.group(1) + m.group(2) if m.group(1).islower() and m.group(2).islower() and len(m.group(1) + m.group(2)) < 15 else m.group(0), text)
    
    # Fix multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Normalize newlines (keep single newlines, remove multiple)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def process_pdfs(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    total_pdfs = len(pdf_files)
    
    if total_pdfs == 0:
        print("  No PDF files found to process.")
        return
    
    processed_count = 0
    skipped_count = 0
    
    for idx, filename in enumerate(pdf_files, 1):
        pdf_path = os.path.join(input_dir, filename)
        txt_path = os.path.join(output_dir, filename.replace(".pdf", ".txt"))
        
        # Check if text file exists and is newer than PDF
        if os.path.exists(txt_path):
            pdf_time = os.path.getmtime(pdf_path)
            txt_time = os.path.getmtime(txt_path)
            if txt_time >= pdf_time:
                skipped_count += 1
                print(f"  Skipping {filename} - already processed ({idx}/{total_pdfs})")
                continue
        
        # Calculate progress percentage (0-25% for PDF processing)
        progress = int((idx / total_pdfs) * 25)
        
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        print(f"  Processing {filename} ({idx}/{total_pdfs}) - {progress}%")
        print(f"    Extracting text from {total_pages} pages...", end="", flush=True)
        
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        # Fix common PDF text extraction issues
        text = fix_pdf_text_spacing(text)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        processed_count += 1
        print(f" ✓ Done")
    
    if skipped_count > 0:
        print(f"  ({skipped_count} file(s) skipped - already up to date)")
