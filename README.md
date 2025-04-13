# 🧠 PaperFinder: Retrieval-Augmented Paper Search

Welcome to **PaperFinder**, a lightweight system for discovering academic papers using semantic search. It leverages **arXiv**, **FAISS**, and **sentence-transformers** to return highly relevant papers for your queries — the foundation for a **Retrieval-Augmented Generation (RAG)** pipeline.

---

## 🚀 Features

- 🔎 Semantic search over academic papers
- 📄 Auto-fetching from [arXiv](https://arxiv.org)
- ⚡ Fast vector similarity with [FAISS](https://github.com/facebookresearch/faiss)
- 🧠 Uses [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for embeddings
- 💬 Chat-style interface (FastAPI + HTML)
- 🔧 RAG-ready backend — just plug in an LLM!

---

## 📸 Demo

![screenshot](static/demo.gif)

---

## 🛠️ Tech Stack

| Layer         | Tool                       |
|---------------|----------------------------|
| API Backend   | FastAPI                    |
| Embeddings    | Sentence Transformers      |
| Search Engine | FAISS                      |
| Paper Source  | arXiv API                  |
| UI            | HTML + JS (Jinja templates) |

---

## ⚙️ Setup Instructions

```bash
# Clone the repo
git clone https://github.com/your-username/paperfinder.git
cd paperfinder

# Set up environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the server
uvicorn main:app --reload
