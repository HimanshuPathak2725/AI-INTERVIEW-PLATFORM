# Backend

FastAPI service for resume parsing, RAG-backed question generation, answer storage, and interview summaries.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/ingest_kb.py --role backend_engineer --file knowledge_base/backend_engineer.txt
python scripts/ingest_kb.py --role ai_ml_engineer --file knowledge_base/ai_ml_engineer.txt
python main.py
```

## Key Endpoints

- `POST /api/v1/sessions`
- `POST /api/v1/sessions/{session_id}/questions`
- `GET /api/v1/sessions/{session_id}/questions`
- `POST /api/v1/answers`
- `POST /api/v1/sessions/{session_id}/complete`
- `GET /api/v1/health`

## Data

- SQLite database: `interview_platform.db`
- Local vector index: `vector_store/kb_index.json`
