from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.staticfiles import StaticFiles

import numpy as np
from pydantic import BaseModel
from typing import Optional, List
import spacy
from sentence_transformers import SentenceTransformer

import arxiv
import faiss

import asyncio

# "python -m spacy download en_core_web_sm" in terminal
nlp = spacy.load("en_core_web_sm")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()
templates = Jinja2Templates(directory="templates") # templates dir
app.mount("/static", StaticFiles(directory="static"), name="static")

# chat history
chat_history = []

# db
papers_db = []
paper_store = []
faiss_index = None
paper_embeddings = None
indexed_papers_ids = set()

class Query(BaseModel):
    query: str
    chat_history: Optional[List[str]] = []

class Paper(BaseModel):
    title: str
    authors: List[str]
    doi: str
    abstract: str

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    # Example: passing a message to the template
    return templates.TemplateResponse("index.html", {"request": request, "message": "Welcome to the Paper Finder!"})

@app.post("/chat")
async def chat(query: Query):
    global chat_history

    user_input = query.query
    chat_history.append(f"User: {user_input}")

    papers = get_papers_from_arvix(user_input)

    if papers:
        papers_db.extend(papers)
        index_papers(papers_db)

    results = await search_papers(user_input)

    sorted_papers = sorted(results['top_papers'], key=lambda x: x['distance'])

    response = "📚 **Top Found Papers:**<br><ul>"
    for i, paper in enumerate(sorted_papers[:3], start=1):
        response += f"<li><strong>{i}. {paper['title']}</strong></li>"
    response += "</ul>"
    
    chat_history.append(f"Assistant: {response}")

    return {
        "response": response,
        "papers": sorted_papers[:3],
        "chat_history": chat_history[-5:] # last 5 turns
    }

def get_papers_from_arvix(query: str, n: int = 10):
    
    search = arxiv.Search(
        query=query,
        max_results=n,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "authors": result.authors,
            "doi": result.doi,
            "abstract": result.summary
        })

    return papers

def encode_papers(papers):
    abstracts = [paper["abstract"] for paper in papers]
    if len(abstracts) == 0:
        return {"error": "No abstracts to encode."}
    embeddings = embed_model.encode(abstracts, convert_to_numpy=True)

    return embeddings

def create_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    print("FAISS index size:", index.ntotal)

    return index

def index_papers(papers):
    global papers_db, paper_store, faiss_index, paper_embeddings, indexed_papers_ids

    # check if the paper exists
    new_papers = [paper for paper in papers if paper["title"] not in indexed_papers_ids]
    if not new_papers:
        return {"message": "No new papers to index"}
    
    for paper in new_papers:
        indexed_papers_ids.add(paper["title"])
    
    papers_db.extend(new_papers)
    new_embeddings = encode_papers(new_papers)

    if faiss_index is None:
        faiss_index = create_faiss_index(new_embeddings)
        paper_embeddings = new_embeddings
    
    else:
        faiss_index.add(new_embeddings)
        paper_embeddings = np.vstack([paper_embeddings, new_embeddings])
    paper_store.extend(new_papers)
    
    print("📌 Total papers in FAISS after indexing:", faiss_index.ntotal)

    return {"message": f"Indexed {len(new_papers)} new papers in FAISS"}

@app.get("/search_papers")
async def search_papers(query: str, k: int = 5):
    global faiss_index, paper_embeddings

    papers = get_papers_from_arvix(query)
    if papers:
        index_papers(papers_db)
    
    query_embedding = embed_model.encode([query], convert_to_numpy = True)
    D, I = faiss_index.search(query_embedding, k)

    top_papers = []
    for j, i in enumerate(I[0]):
        paper = paper_store[i]
        top_papers.append({
            "title": paper["title"],
            "authors": [author.name for author in paper["authors"]],
            "abstract": paper["abstract"],
            "doi": paper["doi"],
            "distance": float(D[0][j])
        })
    
    return {"query": query, "top_papers": top_papers}