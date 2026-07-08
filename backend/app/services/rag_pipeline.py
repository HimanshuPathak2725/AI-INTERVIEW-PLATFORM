"""
RAG pipeline – ingests PDF/TXT knowledge-base files and generates
LLM-grounded interview questions via the GROQ API.
Optimized with asynchronous task handoffs and thread-safe data access layers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import asyncio
import aiofiles
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from app.core.config import Settings

# Conditional FAISS import to prevent crash if running outside sandbox
try:
    import faiss
except ImportError:
    faiss = None

log = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.]{1,}")

def _tokens(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]

def _concept_tokens(text: str) -> set[str]:
    return {
        tok[:-1] if tok.endswith("s") and len(tok) > 4 else tok
        for tok in _tokens(text)
    }

_JINA_MODEL = "jina-embeddings-v3"
_JINA_URL   = "https://api.jina.ai/v1/embeddings"

_ROLE_PATTERNS: dict[str, list[str]] = {
    "ai_ml_engineer": [
        r"machine[\s\-]?learning",
        r"deep[\s\-]?learning",
        r"\bml\b",
        r"\bai\b",
        r"data\s+scientist",
        r"nlp\b",
        r"natural\s+language",
        r"computer\s+vision",
    ],
    "frontend_engineer": [
        r"front[\s\-]?end",
        r"\breact\b",
        r"\bvue\b",
        r"\bangular\b",
    ],
    "backend_engineer": [
        r"back[\s\-]?end",
        r"\bapi\s+engineer\b",
        r"\bdata\s+engineer\b",
        r"\bdevops\b",
        r"\bfullstack\b",
        r"\bsoftware\s+engineer\b",
    ],
}

_COMPILED_ROLE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    role: [re.compile(pat, re.IGNORECASE) for pat in pats]
    for role, pats in _ROLE_PATTERNS.items()
}

class RAGPipeline:
    def __init__(self, backend_root: Optional[Path] = None) -> None:
        self._lock = asyncio.Lock()

        self.backend_root: Path = backend_root or Path(__file__).resolve().parents[2]
        self.kb_root:      Path = self.backend_root / "knowledge_base"
        self.vector_dir:   Path = self.backend_root / "vector_store"

        self.index_path:      Path = self.vector_dir / "kb_index.json"
        self.embeddings_path: Path = self.vector_dir / "kb_embeddings.npy"

        self._chunks:     list[dict]        = []
        self._embeddings: np.ndarray | None = None

        self._role_indexes:    dict[str, faiss.Index] = {}
        self._role_chunk_map:  dict[str, list[int]]   = {}

        self.embedding_dim: int | None = None
        settings = Settings()
        self._jina_key: str = getattr(settings, "JINA_API_KEY", "")
        if not self._jina_key:
            log.warning("[RAG] JINA_API_KEY missing – embeddings will use fallbacks.")

        # Trigger background initialization to prevent blocking constructor
        asyncio.create_task(self._load_index_async())

    def _role_index_path(self, role_key: str) -> Path:
        return self.vector_dir / f"kb_{role_key}.index"

    def _role_key(self, role: str) -> str:
        normalised = re.sub(r"[-_]", " ", role.lower()).strip()
        for key, compiled_pats in _COMPILED_ROLE_PATTERNS.items():
            for pat in compiled_pats:
                if pat.search(normalised):
                    return key
        return "backend_engineer"

    def _fallback_vector(self) -> list[float]:
        dim = self.embedding_dim or 1024
        return [0.0] * dim

    async def _embed_batch_async(self, text: list[str], retries: int = 3) -> list[list[float]]:
        if not self._jina_key or not text:
            return [self._fallback_vector() for _ in text]

        import httpx
        headers = {
            "Authorization": f"Bearer {self._jina_key}",
            "Content-Type":  "application/json",
        }
        
        all_vectors = []
        batch_size = 50
        
        async with httpx.AsyncClient() as client:
            for i in range(0, len(text), batch_size):
                batch_texts = text[i:i+batch_size]
                payload = {"model": _JINA_MODEL, "input": batch_texts}
        
                for attempt in range(retries):
                    try:
                        resp = await client.post(_JINA_URL, headers=headers, json=payload, timeout=60.0)
                        resp.raise_for_status()
                        res_data = resp.json()["data"]
                        vectors = [item["embedding"] for item in res_data]
        
                        if self.embedding_dim is None and vectors:
                            self.embedding_dim = len(vectors[0])
                        all_vectors.extend(vectors)
                        break
                    except Exception as exc:
                        wait = 2 ** attempt
                        log.warning("[RAG] Embedding failed, retrying in %ds: %s", wait, exc)
                        if attempt < retries - 1:
                            await asyncio.sleep(wait)
                else:
                    all_vectors.extend([self._fallback_vector() for _ in batch_texts])
                    
        return all_vectors

    async def _read_file_async(self, path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            return await asyncio.to_thread(self._read_pdf_sync, path)
        async with aiofiles.open(path, mode='r', encoding='utf-8', errors='ignore') as f:
            return await f.read()

    def _read_pdf_sync(self, path: Path) -> str:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            log.error("[RAG] PyPDF2 dependency missing.")
            return ""
        try:
            reader = PdfReader(str(path))
            parts = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(parts)
        except Exception as e:
            log.error("[RAG] Failed reading PDF %s: %s", path.name, e)
            return ""

    def _chunk_text(self, text: str, max_words: int = 400, overlap: int = 40) -> list[str]:
        words = text.split()
        if not words:
            return []
        step = max(max_words - overlap, 1)
        return [" ".join(words[start: start + max_words]).strip() for start in range(0, len(words), step) if words[start: start + max_words]]

    def _rebuild_role_chunk_map(self) -> dict[str, list[int]]:
        role_map: dict[str, list[int]] = {}
        for global_idx, chunk in enumerate(self._chunks):
            role_map.setdefault(chunk["role"], []).append(global_idx)
        return role_map

    async def _build_faiss_async(self) -> None:
        if faiss is None or self._embeddings is None or len(self._embeddings) == 0:
            return

        role_map = self._rebuild_role_chunk_map()
        new_indexes: dict[str, faiss.Index] = {}
        self.vector_dir.mkdir(parents=True, exist_ok=True)

        def _build_sync():
            for role_key, global_indices in role_map.items():
                try:
                    vectors = self._embeddings[global_indices].astype(np.float32).copy()
                    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                    vectors = np.divide(vectors, norms, out=vectors, where=norms > 0)
                    
                    dim = vectors.shape[1]
                    role_idx = faiss.IndexFlatIP(dim)
                    role_idx.add(vectors)
                    new_indexes[role_key] = role_idx
                    faiss.write_index(role_idx, str(self._role_index_path(role_key)))
                except Exception as e:
                    log.error("[RAG] Error building index for %s: %s", role_key, e)

        await asyncio.to_thread(_build_sync)
        async with self._lock:
            self._role_indexes = new_indexes
            self._role_chunk_map = role_map

    async def _load_index_async(self) -> None:
        async with self._lock:
            if not self.index_path.exists():
                return
            try:
                async with aiofiles.open(self.index_path, mode='r', encoding='utf-8') as f:
                    self._chunks = json.loads(await f.read())
                
                if self.embeddings_path.exists():
                    self._embeddings = np.load(self.embeddings_path, allow_pickle=False)
                    if len(self._embeddings):
                        self.embedding_dim = self._embeddings.shape[1]
                
                role_chunk_map = self._rebuild_role_chunk_map()
                if faiss is not None:
                    for role_key in _ROLE_PATTERNS:
                        p = self._role_index_path(role_key)
                        if p.exists():
                            self._role_indexes[role_key] = faiss.read_index(str(p))
                            self._role_chunk_map[role_key] = role_chunk_map.get(role_key, [])
            except Exception as exc:
                log.error("[RAG] Index load aborted, resetting indices: %s", exc)

    async def ingest_document_async(self, file_path: str, role: str) -> bool:
        """
        Schedules document processing into an asynchronous worker task thread 
        so that the incoming HTTP transmission request thread never experiences block latency.
        """
        path = Path(file_path)
        if not path.exists():
            return False

        # Instantly hand-off processing loop to an independent asynchronous task lifecycle
        asyncio.create_task(self._process_ingestion_job(path, role))
        return True

    async def _process_ingestion_job(self, path: Path, role: str) -> None:
        try:
            raw_text = await self._read_file_async(path)
            role_key = self._role_key(role)
            chunks = self._chunk_text(raw_text)
            
            if not chunks:
                return

            embeddings = await self._embed_batch_async(chunks)
            
            async with self._lock:
                new_chunks = []
                base_idx = len(self._chunks)
                for idx, chunk_text in enumerate(chunks):
                    chunk_id = hashlib.sha256(f"{role_key}:{path.name}:{base_idx + idx}".encode()).hexdigest()
                    new_chunks.append({
                        "id":          chunk_id,
                        "role":        role_key,
                        "source":      path.name,
                        "chunk_index": base_idx + idx,
                        "content":     chunk_text,
                    })
                
                self._chunks.extend(new_chunks)
                new_np = np.array(embeddings, dtype=np.float32)
                self._embeddings = new_np if self._embeddings is None else np.vstack([self._embeddings, new_np])
                
                # Non-blocking, isolated persistent storage writer threads
                self.vector_dir.mkdir(parents=True, exist_ok=True)
                tmp_index = self.index_path.with_suffix(".tmp.json")
                async with aiofiles.open(tmp_index, mode='w', encoding='utf-8') as f:
                    await f.write(json.dumps(self._chunks, ensure_ascii=False))
                tmp_index.replace(self.index_path)
                
                await asyncio.to_thread(np.save, self.embeddings_path, self._embeddings, allow_pickle=False)
            
            await self._build_faiss_async()
            log.info("[RAG] Async background processing completed for file: %s", path.name)
        except Exception as e:
            log.error("[RAG] Background processing failed for %s: %s", path.name, e)

    def evaluate_answer(self, question: str, answer: str, expected_concepts: List[str]) -> Dict:
        if not answer.strip():
            return {"score": 0.0, "feedback": "No answer provided.", "concepts_covered": [], "concepts_missing": expected_concepts}

        answer_tokens   = _concept_tokens(answer)
        question_tokens = _concept_tokens(question)

        covered = [c for c in expected_concepts if _concept_tokens(c) & answer_tokens]
        missing = [c for c in expected_concepts if c not in covered]
        relevance = len(answer_tokens & question_tokens) / max(len(question_tokens), 1)
        coverage  = len(covered) / max(len(expected_concepts), 1) if expected_concepts else 1.0
        depth     = min(len(answer.split()) / 70.0, 1.0)
        score     = round((coverage * 55) + (depth * 30) + (relevance * 15), 2)

        return {
            "score": score,
            "feedback": "Strong technical concept articulation." if score >= 75 else "Elaborate with concrete architectural systems design definitions.",
            "concepts_covered": covered,
            "concepts_missing": missing
        }

_pipeline_instance: RAGPipeline | None = None

def get_rag_pipeline() -> RAGPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance
