import os
import json
import pypdf
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

KB_DIR = "knowledgebase"
OUTPUT_FILE = "docs/search_index.json"

# Highly efficient, lightweight embedding model matching the frontend framework
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == ".pdf":
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    text += (page.extract_text() or "") + "\n"
        elif ext in [".html", ".htm"]:
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                text = soup.get_text(separator="\n")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text

def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 30):  # 30-word overlap
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def main():
    if not os.path.exists(KB_DIR):
        os.makedirs(KB_DIR)
        print(f"Created empty '{KB_DIR}' directory. Drop your files there!")
        return

    database = []
    chunk_id = 0
    
    for filename in os.listdir(KB_DIR):
        file_path = os.path.join(KB_DIR, filename)
        if os.path.isfile(file_path) and not filename.startswith('.'):
            print(f"Indexing: {filename}...")
            raw_text = extract_text(file_path)
            chunks = chunk_text(raw_text)
            
            if not chunks:
                continue
                
            # Pre-compute vectors during Python build to save user browser CPU
            embeddings = model.encode(chunks).tolist()
            
            for chunk, vector in zip(chunks, embeddings):
                database.append({
                    "id": chunk_id,
                    "source": filename,
                    "text": chunk,
                    "vector": vector
                })
                chunk_id += 1
                
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(database, f, indent=2)
    print("Database index successfully built!")

if __name__ == "__main__":
    main()
