# Intelligent Support Ticket Resolution Assistant

> A semantic RAG-based support resolution assistant for telecom customer support.

An intelligent support assistant that helps customer-support agents resolve tickets by understanding the meaning of a complaint rather than relying only on keyword matching.

Given a raw customer complaint, the system:

1. Identifies the **intent/category, product, severity, and customer sentiment**
2. Retrieves semantically similar **historical resolved tickets** and **knowledge-base articles**
3. Uses the retrieved evidence to generate a **grounded, step-by-step resolution**
4. Provides **source citations** so agents can trace recommendations back to the evidence used
5. Supports evaluation and iterative improvement as ticket data and classes evolve

---

## 🎥 See It in Action

### ⭐ End-to-End Product Demo

<!-- Insert Product Demo Video Here -->

Demonstrates the complete product workflow — from submitting a raw customer complaint through query analysis, semantic retrieval, grounded resolution generation, and source citations.

### 🛠️ Backend Debugging & Engineering Walkthrough

<!-- Insert Backend Walkthrough Video Here -->

Covers the backend implementation and debugging process, including API flow, query analysis, retrieval, LLM integration, validation, fallback handling, and testing.

---

## 📌 Problem Statement

Telecom customer-support agents often search historical tickets and knowledge bases using keywords such as:

- `router`
- `billing`
- `internet`

This approach is limited because customers can describe the same underlying problem using very different words.

For example:

> **"My broadband drops every evening around 8 and I've already restarted the router twice. I work from home and this is costing me."**

A keyword-based system may focus on terms such as `broadband` or `router`.

A semantic system can instead understand the broader problem:

- recurring connectivity failure
- broadband service
- router already restarted
- repeated issue
- customer frustration
- potentially high severity due to work disruption

The objective is to move from **keyword-based ticket search** to **meaning-based resolution assistance**.

---

# 🧠 Solution Overview

The system follows a Retrieval-Augmented Generation (RAG) architecture.

```text
                         ┌──────────────────────┐
                         │   Support Agent      │
                         │  Raw Customer Query  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Query Analyzer     │
                         │                      │
                         │ • Intent / Category  │
                         │ • Product            │
                         │ • Severity           │
                         │ • Sentiment          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Semantic Retrieval   │
                         │                      │
                         │ Historical Tickets   │
                         │ Knowledge Base       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Retrieved Context  │
                         │                      │
                         │ Similar Tickets      │
                         │ KB Articles          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      LLM / RAG       │
                         │                      │
                         │ Grounded Resolution  │
                         │ Generation           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Validation        │
                         │                      │
                         │ • Output structure   │
                         │ • Grounding          │
                         │ • Fallback handling  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Agent UI         │
                         │                      │
                         │ Resolution + Sources │
                         └──────────────────────┘
```

---

# 🏗️ Architecture

The application is organized as a lightweight microservice-oriented system.

### Frontend

The frontend provides the support-agent interface for:

- Entering a customer complaint
- Submitting a resolution request
- Viewing interpreted ticket attributes
- Viewing retrieved evidence
- Viewing the generated resolution
- Inspecting source citations

Built using **React + Vite**.

### Backend

The backend exposes the application API and orchestrates the resolution pipeline.

Core responsibilities include:

- Request validation
- Query analysis
- Hybrid semantic & lexical retrieval
- Context construction
- LLM interaction (Gemini with Groq fallback)
- Response validation and grounding checks
- Error and fallback handling

The backend is implemented using **FastAPI**.

### Query Analysis

The incoming customer complaint is transformed into structured information:

```text
Intent / Category
Product
Severity
Customer Sentiment
```

This provides a richer representation of the customer's problem for downstream retrieval and generation.

### Semantic Retrieval

Instead of relying only on lexical overlap, the system uses embeddings (`BAAI/bge-base-en-v1.5`) alongside PostgreSQL `pgvector` to represent the meaning of the complaint and retrieve semantically similar information.

The retrieval layer searches across:

- Historical resolved support tickets
- Knowledge-base content

Historical resolutions provide practical evidence about how similar problems were previously handled.

### Retrieval-Augmented Generation

Retrieved information is supplied as context to the LLM.

The model generates a resolution based on the retrieved evidence rather than freely inventing an answer.

```text
Customer Complaint
        +
Query Analysis
        +
Retrieved Historical Resolutions
        +
Knowledge Base Evidence
        ↓
Grounded Resolution
        +
Source Citations
```

---

# 📂 Project Structure

```text
Intelligent-Support-Ticket-Resolution-Assistant/
│
├── app/
│   ├── __init__.py
│   ├── analyzer.py
│   ├── config.py
│   ├── database.py
│   ├── embeddings.py
│   ├── ingestion.py
│   ├── llm.py
│   ├── main.py
│   ├── pipeline.py
│   ├── rag.py
│   ├── retrieval.py
│   ├── schemas.py
│   └── validation.py
│
├── data/
│   ├── evaluation/
│   │   ├── retrieval_eval.json
│   │   └── retrieval_results.json
│   │
│   └── processed/
│       ├── eval_tickets_100.json
│       ├── telecom_historical_clean.json
│       ├── telecom_historical_test.json
│       ├── telecom_kb.json
│       └── tickets_5000.json
│
├── frontend/
│   ├── public/
│   └── src/
│       ├── App.jsx
│       ├── App.css
│       ├── index.css
│       └── main.jsx
│
├── scripts/
│   ├── clean_telecom_corpus.py
│   ├── evaluate_retrieval.py
│   ├── ingest_kb.py
│   ├── ingest_telecom_historical.py
│   ├── prepare_dataset.py
│   ├── prepare_telecom_corpus.py
│   └── rebuild_embeddings.py
│
├── sql/
│   └── schema.sql
│
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py
│   ├── test_rag_error.py
│   ├── test_rag_fallback.py
│   └── test_validation.py
│
├── docker-compose.yml
├── requirements.txt
└── .gitignore
```

---

# 📊 Dataset Description

The system uses processed telecom support data representing historical customer-support interactions and knowledge-base information.

### Historical Tickets

`data/processed/telecom_historical_clean.json`

Cleaned historical support-ticket data used as a primary source for semantic retrieval and resolution patterns.

### Historical Test Data

`data/processed/telecom_historical_test.json`

Held-out historical ticket data used for evaluation and testing.

### Knowledge Base

`data/processed/telecom_kb.json`

Knowledge-base content used alongside historical tickets during retrieval.

### Support Ticket Dataset

`data/processed/tickets_5000.json`

A larger processed support-ticket collection used during development and experimentation.

### Evaluation Dataset

`data/processed/eval_tickets_100.json`

A dedicated evaluation set containing 100 tickets for assessing system behaviour.

---

# 🔬 Evals on System Health

A production-oriented RAG system should not be evaluated only by whether the final answer looks good.

The project includes a dedicated retrieval evaluation workflow.

```text
Evaluation Dataset
        │
        ▼
   Query Pipeline
        │
        ▼
 Semantic Retrieval
        │
        ▼
 Compare Retrieved
 Results Against
 Expected Evidence
        │
        ▼
 Retrieval Metrics
```

Evaluation artifacts are maintained under:

```text
data/evaluation/
├── retrieval_eval.json
└── retrieval_results.json
```

The retrieval evaluation workflow is implemented in:

```text
scripts/evaluate_retrieval.py
```

### Evaluation areas

#### Retrieval quality

- Top-K retrieval performance
- Relevant-ticket retrieval
- Knowledge-base retrieval quality
- Ranking quality

#### Resolution quality

- Groundedness of generated responses
- Correct use of retrieved evidence
- Citation correctness
- Resolution completeness

#### System health

- API latency
- Retrieval latency
- LLM latency
- Failure rates
- Fallback frequency

#### Regression behaviour

The evaluation workflow can be rerun when:

- The embedding model changes
- Retrieval logic changes
- Query analysis changes
- New ticket classes are introduced
- The knowledge base is updated
- Prompting or LLM configuration changes

---

# 🔎 Additional Exploration

## 1. Semantic Search vs Keyword Search

The core motivation is to evaluate whether semantic retrieval can identify relevant historical resolutions even when the customer's wording differs substantially from previous tickets.

```text
Historical ticket:
"Internet connection disconnects every evening."

New complaint:
"My broadband keeps dropping around 8 PM every day."
```

The wording is different, but the underlying problem is similar.

## 2. Historical Resolution Reuse

Historical tickets are not treated merely as search results; their previous resolution steps provide operational knowledge that can be reused when generating a recommendation for a new complaint.

## 3. Knowledge Base + Historical Tickets

Using both sources allows the system to combine **structured knowledge** from knowledge-base articles with **empirical resolution patterns** from previously resolved tickets.

## 4. Evolving Ticket Classes

Support data changes over time. The ingestion and retrieval pipeline supports adding new ticket data and categories without requiring the entire application architecture to be rewritten.

## 5. Fallback & Error Handling

The project includes validation and fallback handling for situations such as:

- Retrieval failure or low confidence
- Invalid model output
- Missing context
- LLM/API errors
- Empty or malformed input

---

# 🚀 Production Scale Considerations

## 1. Vector Index Scaling

The retrieval layer leverages PostgreSQL with `pgvector` and can scale to dedicated distributed vector databases (e.g., Qdrant, Milvus, Pinecone, or OpenSearch) as corpus size grows.

## 2. Incremental Data Ingestion

The ingestion scripts allow newly resolved tickets and KB articles to be embedded and added incrementally without full corpus re-indexing.

## 3. Model Versioning & Observability

Tracking embedding models, LLM versions, prompt templates, retrieval latency, token usage, and resolution acceptance by human support agents ensures repeatable evals and regression monitoring.

## 4. Security & Human-in-the-Loop

Sensitive customer information can be scrubbed for PII before indexing. Low-confidence resolutions or uncited recommendations trigger automatic escalation to human agents.

---

# ⚙️ Getting Started

## Prerequisites

- Python 3.10+
- Node.js 18+ / npm
- Docker (for PostgreSQL + pgvector)
- API key for Gemini (and optionally Groq for fallback)

---

## 1. Database Setup

Start PostgreSQL with `pgvector` using Docker:

```bash
docker compose up -d
```

Initialize the database schema:

```bash
psql -h localhost -U support_user -d support_db -f sql/schema.sql
```

*(Default password: `support_password`)*

---

## 2. Backend Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://support_user:support_password@localhost:5432/support_db
GEMINI_API_KEY=your_gemini_api_key
LLM_MODEL=gemini-3.8-flash

GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b

APP_NAME=Intelligent Support Ticket Resolution Assistant
APP_ENV=development
```

Ingest the knowledge base and historical ticket embeddings:

```bash
python -m scripts.ingest_kb
python -m scripts.ingest_telecom_historical
```

Start the FastAPI application:

```bash
uvicorn app.main:app --reload
```

The backend will be available at `http://localhost:8000` (API docs at `http://localhost:8000/docs`).

---

## 3. Frontend Setup

In a separate terminal, navigate to the frontend directory:

```bash
cd frontend
npm install
npm run dev
```

The frontend will run at `http://localhost:5173`.

---

# 🧪 Running Tests & Evals

Run the test suite:

```bash
pytest
```

Run retrieval evaluation:

```bash
python -m scripts.evaluate_retrieval
```

---

# 🎯 Key Takeaway

> **Don't search for the same words. Search for the same problem.**

By combining query understanding, semantic retrieval, historical resolution patterns, knowledge-base evidence, and grounded LLM generation, the assistant helps support agents move from manually searching past tickets to receiving an evidence-backed resolution workflow.
