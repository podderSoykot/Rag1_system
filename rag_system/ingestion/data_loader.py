# rag_system/ingestion/data_loader.py

import os
from pypdf import PdfReader

def process_pdfs(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            txt_path = os.path.join(output_dir, filename.replace(".pdf", ".txt"))

            reader = PdfReader(pdf_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"✅ {filename} → {txt_path}")
