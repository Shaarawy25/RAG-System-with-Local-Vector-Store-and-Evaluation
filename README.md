
# 🧩 Retrieval-Augmented Generation (RAG) System

---

## 📦 Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system using open-source tools and local vector storage. It enhances Large Language Models (LLMs) by retrieving relevant information from an external knowledge base before generating responses. The system is modular and fully configurable to support experimentation with:

* Different Embedding Models
* Retrieval Strategies
* Prompt Templates
* Evaluation Metrics

---

## 🚀 Setup Instructions

1. **Clone the Repository:**

```bash
git clone <your-repository-url>
cd RAG_System
```

2. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

3. **Environment Configuration:**
   Create a `.env` file in the root directory with the following content:

```
GROQ_API_KEY=your_actual_api_key
GROQ_MODEL=mixtral-8x7b-32768
MODEL_SERVER=GROQ
TOKENIZERS_PARALLELISM=false
```

4. **Add Your Documents:**
   Place all `.pdf`, `.docx`, and `.txt` files into the `documents/` folder.

5. **Generate Embeddings and Vector Store:**

```bash
python -m app.main_entry
```

6. **Run Evaluation:**

```bash
python -m app.evaluation
```

7. **Manual Query Interface:**

```bash
python -m app.rag_pipeline
```

---

## 📚 RAG System Architecture

```
Input Query
     │
     ▼
Document Loader (PDF, DOCX, TXT)
     │
     ▼
Document Chunking & Embedding (FAISS + Sentence Transformers)
     │
     ▼
Retrieval (Basic Similarity / MMR / Hybrid Search)
     │
     ▼
Prompt Construction (Standard / QA Template)
     │
     ▼
LLM Generation via LangChain & Groq API
     │
     ▼
Final Answer
```

---

### 🔧 Components

* **Embedding Models:**

  * `all-MiniLM-L6-v2`
  * `paraphrase-MiniLM-L3-v2`

* **Retrieval Strategies:**

  * Basic Similarity Search
  * Maximum Marginal Relevance (MMR)
  * Hybrid Search

* **Prompt Templates:**

  * Standard
  * QA Template

---

## 📊 Experimental Results

| Embedding Model         | Retrieval Strategy | Prompt Template | Precision | Recall | F1-Score |
| ----------------------- | ------------------ | --------------- | --------- | ------ | -------- |
| all-MiniLM-L6-v2        | Basic Similarity   | Standard        | 1.0       | 1.0    | 1.0      |
| all-MiniLM-L6-v2        | MMR                | QA Template     | 1.0       | 1.0    | 1.0      |
| paraphrase-MiniLM-L3-v2 | Hybrid             | Standard        | 1.0       | 1.0    | 1.0      |
| paraphrase-MiniLM-L3-v2 | MMR                | QA Template     | 1.0       | 1.0    | 1.0      |

> ⚠️ These perfect scores suggest the current semantic similarity threshold may be too lenient. For more realistic results, try increasing the threshold in `evaluation.py`.

---

## 📈 Evaluation Metrics

* **Precision**: Correctly relevant results among retrieved.
* **Recall**: Correctly relevant results among all possible relevant.
* **F1-Score**: Harmonic mean of Precision and Recall.

---

## ✅ Strengths of the Approach

* Fully modular and configurable system.
* Supports multiple retrieval and embedding strategies.
* Easy integration with Groq API via LangChain.
* Automatic evaluation and comparison framework.
* Clean, extensible architecture.

---

## ⚠️ Weaknesses of the Approach

* Evaluation is currently dependent on a semantic similarity threshold, which may produce inflated metrics.
* Prompt templates are basic; could benefit from further engineering.
* No advanced post-retrieval filtering applied (Bonus Challenge left incomplete).

---

## 📌 Challenges Encountered & Solutions

| Challenge                      | Solution                                                |
| ------------------------------ | ------------------------------------------------------- |
| Incorrect API URL for Groq     | Switched to LangChain's Groq integration.               |
| FAISS Index Path Errors        | Standardized paths for all executions.                  |
| Overly Strict Evaluation Logic | Replaced with semantic similarity checking.             |
| Hardcoded Queries              | Introduced interactive CLI and parameterized pipelines. |

---

## 📄 Document Corpus

You can use any AI-related research papers or documents. Suggested sources:

* [arXiv.org](https://arxiv.org/)
* [Papers with Code](https://paperswithcode.com/)

Place downloaded files in the `documents/` folder.

---

## 📁 Configuration Files & Saved States

* **.env**: Contains API keys and model configurations.
* **saved\_vectorstores/**: Stores FAISS index and metadata.

  * `index.faiss`: FAISS vector store.
  * `metadata.pkl`: Chunk metadata.
