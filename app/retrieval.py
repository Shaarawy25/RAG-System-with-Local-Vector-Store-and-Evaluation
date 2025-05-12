import faiss
import numpy as np
import pickle

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def load_metadata(metadata_path):
    with open(metadata_path, 'rb') as f:
        return pickle.load(f)

def basic_similarity_search(query_embedding, index, metadata, top_k=5):
    D, I = index.search(np.array([query_embedding]).astype('float32'), top_k)
    results = []
    for idx in I[0]:
        results.append(metadata[idx])
    return results

def mmr_search(query_embedding, index, metadata, top_k=5, lambda_param=0.5):
    # Placeholder for real MMR implementation. For now, return basic similarity search.
    return basic_similarity_search(query_embedding, index, metadata, top_k)

def hybrid_search(query_embedding, keyword_matches, index, metadata, top_k=5):
    semantic_results = basic_similarity_search(query_embedding, index, metadata, top_k)
    # Simple hybrid by merging results and removing duplicates
    combined = {chunk['content']: chunk for chunk in semantic_results + keyword_matches}
    return list(combined.values())[:top_k]

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    index = load_faiss_index("saved_vectorstores/index.faiss")
    metadata = load_metadata("saved_vectorstores/metadata.pkl")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query = "Explain Retrieval-Augmented Generation."
    query_embedding = model.encode(query)

    results = basic_similarity_search(query_embedding, index, metadata, top_k=3)
    for res in results:
        print(f"Source: {res['metadata']['source']}, Content: {res['content'][:200]}...")
