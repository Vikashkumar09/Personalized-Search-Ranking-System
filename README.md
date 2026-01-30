# Personalized-Search-Ranking-System
This project implements an end-to-end semantic search and ranking pipeline using modern NLP models and scalable vector search. It combines dense embeddings, approximate nearest neighbor retrieval, neural re-ranking, and learning-to-rank techniques, and exposes the system via a FastAPI service

Key Features

Semantic Embeddings
Uses sentence-transformers/all-MiniLM-L6-v2 to generate dense vector representations for documents and queries.

Scalable Vector Search (FAISS)
Builds a FAISS HNSW index for fast approximate nearest neighbor (ANN) retrieval, with optional GPU acceleration.

Two-Stage Ranking Pipeline

Candidate Retrieval via embedding similarity

Cross-Encoder Re-Ranking using cross-encoder/ms-marco-MiniLM-L-6-v2 for higher precision results

Learning-to-Rank (LTR)
Demonstrates ranking evaluation (NDCG, MRR) and integrates an XGBoost LambdaRank model for supervised ranking experiments.

API Deployment
Exposes search functionality through a FastAPI endpoint, making the system easy to deploy and integrate into applications.


Tech Stack

Python, PyTorch

Hugging Face Transformers

FAISS (CPU/GPU)

XGBoost (LambdaRank)

FastAPI + Uvicorn

Pandas, NumPy, scikit-learn
