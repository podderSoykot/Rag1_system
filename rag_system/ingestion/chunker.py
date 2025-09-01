# rag_system/ingestion/chunker.py

import os

def process_files(input_dir, output_dir, chunk_size=500):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".txt", "_chunks.txt"))

            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

            with open(output_path, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(chunk.replace("\n", " ") + "\n")

            print(f"✅ {filename} → {len(chunks)} chunks")
