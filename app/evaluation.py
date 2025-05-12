import os
from sklearn.metrics import precision_score, recall_score, f1_score
from langchain_groq import ChatGroq
from .rag_pipeline import rag_query_pipeline
from .retrieval import basic_similarity_search, mmr_search
from sentence_transformers import SentenceTransformer, util

def calculate_metrics(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

def is_semantically_relevant(expected, generated, threshold=0.7):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([expected, generated], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return int(similarity >= threshold)

def evaluate_rag_system(api_key, test_cases, embedding_models, retrieval_strategies, prompt_templates):
    results_summary = []

    for embed_model in embedding_models:
        for retrieval_method in retrieval_strategies:
            for prompt_template in prompt_templates:

                y_true = []
                y_pred = []

                print(f"\n=== Configuration: Embedding={embed_model}, Retrieval={retrieval_method.__name__}, PromptTemplate={prompt_template} ===")

                for case in test_cases:
                    query = case["query"]
                    expected = case["expected_answer"].lower()

                    response = rag_query_pipeline(query, api_key, embed_model, retrieval_method, prompt_template).lower()

                    is_relevant = is_semantically_relevant(expected, response)
                    y_pred.append(is_relevant)
                    y_true.append(1)

                    print(f"\nQuery: {query}\nExpected: {expected}\nGenerated: {response}\nRelevant: {is_relevant}")

                metrics = calculate_metrics(y_true, y_pred)

                results_summary.append({
                    "Embedding Model": embed_model,
                    "Retrieval Strategy": retrieval_method.__name__,
                    "Prompt Template": prompt_template,
                    **metrics
                })

    print("\n--- Final Comparison Report ---")
    for result in results_summary:
        print(result)

if __name__ == "__main__":
    api_key = os.getenv("GROQ_API_KEY")

    test_cases = [
        {"query": "What is Retrieval-Augmented Generation?", "expected_answer": "Retrieval-Augmented Generation is a method to enhance language model outputs by retrieving relevant external knowledge before generating a response."},
        {"query": "Explain the concept of transformers.", "expected_answer": "Transformers are neural network architectures used primarily in NLP tasks, characterized by self-attention mechanisms."}
    ]

    embedding_models = ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2"]
    retrieval_strategies = [basic_similarity_search, mmr_search]
    prompt_templates = ["standard", "qa_template"]

    evaluate_rag_system(api_key, test_cases, embedding_models, retrieval_strategies, prompt_templates)