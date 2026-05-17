# 🛍️ Flipkart Product Recommendation Chatbot

An AI-powered intelligent product recommendation system built with hybrid semantic + keyword search and a conversational chatbot interface.

> **Internship Project** | B.Tech CSE (Hons.) — Data Science & Data Engineering  
> Lovely Professional University, Phagwara, Punjab | Course Code: CSE461  
> **Student:** V. L. N. Naga Sathvik (Reg: 12211596) | **Supervisor:** Mr. Madhav Dubey (UID: 65167)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Methodology](#methodology)
- [Results](#results)
- [Limitations](#limitations)
- [Future Scope](#future-scope)
- [References](#references)

---

## Overview

Online shopping platforms like Flipkart offer thousands of products, creating **decision overload** for users. This project tackles that by building an intelligent recommendation system that understands *what users mean*, not just *what they type*.

Instead of basic keyword search, the system uses:
- **NLP** to interpret user intent
- **Vector embeddings** to understand semantic meaning
- **Hybrid search** (FAISS + BM25) for accurate retrieval
- **LLM-powered chatbot** for natural conversation

**Example interaction:**
```
User:   "Show me good phones under 10000"
System: "Here are some options. Do you prefer better battery or camera?"
```

---

## Problem Statement

Traditional e-commerce search has key shortcomings:

| Problem | Description |
|---|---|
| Keyword-only search | Matches words, not meaning |
| No personalization | Every user gets the same results |
| No conversation | Users can't refine queries naturally |
| Cold start | Hard to recommend for new users/products |

---

## Features

- ✅ Natural language query understanding
- ✅ Semantic search via FAISS (meaning-based)
- ✅ Keyword search via BM25 (exact-match)
- ✅ Hybrid ranking for best-of-both accuracy
- ✅ Conversational chatbot with session memory
- ✅ LLM-generated explanations for recommendations
- ✅ Fast response time, scalable architecture

---

## System Architecture

```
User Query
    │
    ▼
NLP Preprocessing
    │
    ├──────────────────┐
    ▼                  ▼
FAISS Search        BM25 Search
(Semantic)          (Keyword)
    │                  │
    └──────┬───────────┘
           ▼
    Hybrid Ranking
           │
           ▼
    LLM Response Generation
           │
           ▼
    Final Recommendations
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Embeddings | Sentence Transformers |
| Semantic Search | FAISS |
| Keyword Search | BM25 |
| Backend API | FastAPI |
| UI | Gradio |
| Chatbot / LLM | Ollama / Groq |

---

## Methodology

1. **Data Collection** — Product data (name, price, category, description, ratings)
2. **Preprocessing** — Clean text, handle missing values, normalize
3. **Feature Engineering** — Combine relevant fields into unified text
4. **Embedding Generation** — Convert text to vectors via Sentence Transformers
5. **FAISS Indexing** — Store vectors for fast similarity search
6. **Query Processing** — Clean and vectorize user input
7. **Dual Search** — Run FAISS (semantic) and BM25 (keyword) in parallel
8. **Hybrid Ranking** — Merge and rank results by relevance + similarity
9. **Response Generation** — LLM explains and presents recommendations

---

## Results

### Semantic Search (FAISS)
- Query: *"Affordable smart watches under 1500"* → returned watches at ₹1299, ₹1399, ₹1499
- Understood "affordable" without exact price keywords

### Keyword Search (BM25)
- High precision for exact terms
- Less flexible for meaning-based queries

### Hybrid Model
- Best overall accuracy
- Combines contextual understanding with keyword precision

### Comparison

| Feature | Traditional System | This System |
|---|---|---|
| Keyword Matching | ✅ | ✅ |
| Semantic Understanding | ❌ | ✅ |
| Personalization | Limited | High |
| Conversational Interface | ❌ | ✅ |

### Response Time
- BM25: Fastest (simple word matching)
- FAISS: Slightly slower (vector computation)
- Hybrid: Slowest but most accurate — acceptable trade-off

---

## Modules

The codebase is split into independent modules:

```
├── engine.py              # Core search engine
├── ranker.py              # Hybrid ranking logic
├── faiss_index.py         # FAISS vector indexing
├── production_pipeline.py # End-to-end pipeline
├── bm25_search.py         # BM25 keyword search
└── chatbot.py             # LLM chatbot interface
```

---

## Limitations

- No real-time data from Flipkart (static dataset only)
- Session-only memory (no long-term personalization)
- Vague queries may yield less relevant results
- Chatbot quality depends on the underlying LLM model
- No payment or cart integration

---

## Future Scope

- 🔗 Real-time Flipkart API integration
- 🎙️ Voice-based query input
- 📱 Android/iOS mobile app
- 🌐 Multi-language support
- 🧠 Advanced personalization using user history
- 💡 Explainable AI — show *why* a product is recommended
- 🛒 Cart and checkout integration
- 🤖 Deep learning models for improved accuracy

---

## References

- Lewis et al. — [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), NeurIPS 2020
- Johnson et al. — [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734), IEEE 2017
- Reimers & Gurevych — [Sentence-BERT](https://arxiv.org/abs/1908.10084), EMNLP 2019
- Karpukhin et al. — [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906), EMNLP 2020
- Khattab & Zaharia — [ColBERT](https://arxiv.org/abs/2004.12832), SIGIR 2020
- Devlin et al. — [BERT](https://arxiv.org/abs/1810.04805), NAACL 2019
- Covington et al. — [Deep Neural Networks for YouTube Recommendations](https://dl.acm.org/doi/10.1145/2959100.2959190), RecSys 2016
- Manning et al. — [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/), Cambridge University Press 2008
