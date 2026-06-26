# AI Interview Platform

AI-powered role-based candidate screening system for the AI/ML and backend intern assignment.

## What It Does

- Parses PDF or pasted-text resumes and extracts skills, technologies, domains, education, and experience hints.
- Builds role-aware retrieval queries from the resume and selected target role.
- Ingests role-specific knowledge base files into a persisted local vector index.
- Retrieves relevant chunks and generates traceable interview questions from that context.
- Stores sessions, questions, answers, scores, feedback, and retrieval context in SQLite.
- Provides a React interview flow with upload, question answering, context trace, and final summary.

## Architecture

- `frontend/`: Vite + React screening console.
- `backend/main.py`: FastAPI app entry point.
- `backend/app/api/routes.py`: Session, question, answer, summary, and health endpoints.
- `backend/app/services/resume_parser.py`: Resume text extraction and profile parsing.
- `backend/app/services/rag_pipeline.py`: Chunking, hashed embeddings, JSON vector index, retrieval, question generation, and answer scoring.
- `backend/app/services/interview_service.py`: Interview lifecycle orchestration.
- `backend/app/models/database.py`: SQLAlchemy persistence models.
- `backend/knowledge_base/`: Role-specific corpus files.

## Quick Start

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/ingest_kb.py --role backend_engineer --file knowledge_base/backend_engineer.txt
python scripts/ingest_kb.py --role ai_ml_engineer --file knowledge_base/ai_ml_engineer.txt
python main.py
```

API health check:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## Design Decisions

- The RAG pipeline uses a local persisted vector index in `backend/vector_store/kb_index.json`. This keeps the project portable and demoable without requiring a cloud API key or native vector database install.
- Embeddings use a deterministic hashed bag-of-words vector. It is lightweight, fast, and sufficient to demonstrate chunking, vector retrieval, query construction, and traceability.
- Generated questions store the retrieved context in the database, so each question can be audited after the session.
- The system works without an OpenAI key. A production version could replace the deterministic question generator with an LLM call after retrieval.

## Deliverables Notes

- Demo video should show knowledge ingestion, starting both servers, uploading or pasting a resume, generating questions, submitting answers, and viewing the summary.
- The included knowledge base files are compact local corpora for demonstration. For a final submission, expand them with notes from the assignment-approved books.
