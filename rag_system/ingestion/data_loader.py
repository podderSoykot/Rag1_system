import os
from pypdf import PdfReader
from pathlib import Path

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
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        processed_count += 1
        print(f" ✓ Done")
    
    if skipped_count > 0:
        print(f"  ({skipped_count} file(s) skipped - already up to date)")
