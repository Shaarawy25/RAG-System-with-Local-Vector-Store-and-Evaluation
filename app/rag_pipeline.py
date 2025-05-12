from .retrieval import load_faiss_index, load_metadata, basic_similarity_search
from sentence_transformers import SentenceTransformer
import requests
import os
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()


def generate_llm_response_groq(prompt):
    groq_api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL")

    chat_model = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
    response = chat_model.invoke(prompt)
    
    return response.content if hasattr(response, 'content') else str(response)



def create_prompt(context_chunks, query):
    context = "\n\n".join([chunk['content'] for chunk in context_chunks])
    prompt = f"""Use the following context to answer the question. 

Context:
{context}

Question: {query}
Answer:"""
    return prompt

def rag_query_pipeline(query, api_key, embedding_model="all-MiniLM-L6-v2", retrieval_method=basic_similarity_search, prompt_template="standard", top_k=5):
    index = load_faiss_index("saved_vectorstores/index.faiss")
    metadata = load_metadata("saved_vectorstores/metadata.pkl")

    model = SentenceTransformer(embedding_model)
    query_embedding = model.encode(query)

    relevant_chunks = retrieval_method(query_embedding, index, metadata, top_k=top_k)
    prompt = create_prompt(relevant_chunks, query)

    llm_response = generate_llm_response_groq(prompt)
    return llm_response


if __name__ == "__main__":
    API_KEY = os.getenv("GROQ_API_KEY")  # Ensure your key is set as an environment variable
    if not API_KEY:
        raise ValueError("API key not found. Please set the GROQ_API_KEY environment variable.")
    query = input("Enter your query: ")
    response = rag_query_pipeline(query,API_KEY)
    print("\nFinal Answer:\n", response)
