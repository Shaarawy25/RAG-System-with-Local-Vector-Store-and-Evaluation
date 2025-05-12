from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

def generate_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = [chunk['content'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def save_faiss_index(embeddings, index_path):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)

def load_faiss_index(index_path):
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    return None

def save_metadata(metadata, path):
    with open(path, 'wb') as f:
        pickle.dump(metadata, f)

def load_metadata(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return []

if __name__ == "__main__":
    from document_loader import load_documents
    from document_processor import split_documents
    import numpy as np

    docs = load_documents("../documents")
    chunks = split_documents(docs)

    embeddings = generate_embeddings(chunks)
    embeddings_np = np.array(embeddings).astype('float32')

    os.makedirs("../saved_vectorstores", exist_ok=True)
    save_faiss_index(embeddings_np, "../saved_vectorstores/index.faiss")
    save_metadata(chunks, "../saved_vectorstores/metadata.pkl")

    print("Embeddings and metadata saved.")