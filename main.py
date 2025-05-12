from app.document_loader import load_documents
from app.document_processor import split_documents
from app.embedding_generator import generate_embeddings, save_faiss_index, save_metadata

import os
import numpy as np

def setup_pipeline():
    docs = load_documents("./documents")
    print(f"Loaded {len(docs)} documents.")

    chunks = split_documents(docs)
    print(f"Generated {len(chunks)} chunks.")

    embeddings = generate_embeddings(chunks)
    embeddings_np = np.array(embeddings).astype('float32')

    os.makedirs("./saved_vectorstores", exist_ok=True)
    save_faiss_index(embeddings_np, "./saved_vectorstores/index.faiss")
    save_metadata(chunks, "./saved_vectorstores/metadata.pkl")

    print("Pipeline setup complete. Embeddings and metadata saved.")

if __name__ == "__main__":
    setup_pipeline()
