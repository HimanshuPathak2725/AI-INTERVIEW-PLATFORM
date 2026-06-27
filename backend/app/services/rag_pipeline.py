"""
RAG pipeline – ingests PDF/TXT knowledge-base files and generates
LLM-grounded interview questions via the GROQ API.

logs
─────────
      - Atomic writes for JSON and NPY to prevent corruption (Fix 6).
      - Partial role-index loading to avoid unnecessary rebuilds (Fix 1).
      - Orphaned FAISS index cleanup when roles are deleted (Fix 2).
      - Zero-vector normalization guards to prevent NaN FAISS poisoning (Fix 4).
      - Fallback dimension validation explicitly preventing corrupt vectors (Fix 3).
      - Secure deserialization `allow_pickle=False` for NumPy (Fix 5).
      - Retrieval bounds clamping `top_k = min(top_k, ntotal)` (Fix 7).
      - Fallback semantic text inspection for file role classification (Fix 8).
      - `typing.Counter` compatibility for Python <3.9 (Fix 9).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Counter as TCounter

import faiss
import numpy as np

log = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.]{1,}")

# ── NLP helpers ───────────────────────────────────────────────────────────────


def _tokens(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def _concept_tokens(text: str) -> set[str]:
    return {
        tok[:-1] if tok.endswith("s") and len(tok) > 4 else tok
        for tok in _tokens(text)
    }


# ── Constants ─────────────────────────────────────────────────────────────────

_JINA_MODEL = "jina-embeddings-v3"
_JINA_URL   = "https://api.jina.ai/v1/embeddings"
_GROQ_MODEL = "llama-3.3-70b-versatile"

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
        r"research\s+scientist",
        r"research\s+engineer",
    ],
    "frontend_engineer": [
        r"front[\s\-]?end",
        r"\breact\b",
        r"\bvue\b",
        r"\bangular\b",
        r"\bui\s+engineer\b",
        r"\bui\s+developer\b",
    ],
    "backend_engineer": [
        r"back[\s\-]?end",
        r"\bapi\s+engineer\b",
        r"\bdata\s+engineer\b",
        r"\bdevops\b",
        r"\binfrastructure\b",
        r"\bfullstack\b",
        r"\bfull[\s\-]?stack\b",
        r"\bserver\s+engineer\b",
        r"\bsoftware\s+engineer\b",
        r"\bsoftware\s+developer\b",
        r"\bpython\s+developer\b",
        r"\bjava\s+developer\b",
        r"\bnode\s+developer\b",
    ],
}

_COMPILED_ROLE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    role: [re.compile(pat, re.IGNORECASE) for pat in pats]
    for role, pats in _ROLE_PATTERNS.items()
}


# ── Main class ────────────────────────────────────────────────────────────────


class RAGPipeline:
    """Ingest PDF/TXT knowledge-base → retrieve context →
    generate LLM-grounded interview questions."""

    def __init__(self, backend_root: Optional[Path] = None) -> None:
        self._lock = threading.RLock()

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

        self._jina_key: str = self._load_secret("JINA_API_KEY")
        if not self._jina_key:
            log.warning("[RAG] JINA_API_KEY missing – embeddings will be zero vectors.")

        self._load_index()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _role_index_path(self, role_key: str) -> Path:
        return self.vector_dir / f"kb_{role_key}.index"

    @staticmethod
    def _load_secret(env_var: str) -> str:
        value = os.getenv(env_var, "")
        if not value:
            try:
                from app.core.config import settings  # type: ignore
                value = getattr(settings, env_var, "") or ""
            except Exception:
                pass
        return value.strip()

    def _role_key(self, role: str) -> str:
        normalised = re.sub(r"[-_]", " ", role.lower()).strip()
        for key, compiled_pats in _COMPILED_ROLE_PATTERNS.items():
            for pat in compiled_pats:
                if pat.search(normalised):
                    return key
        return "backend_engineer"

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _fallback_vector(self) -> list[float]:
        # Fix 3: Fallback dimension trap
        if self.embedding_dim is None:
            raise RuntimeError(
                "Cannot generate fallback embeddings before the embedding dimension is established."
            )
        return [0.0] * self.embedding_dim

    def _embed(self, text: str, retries: int = 3) -> list[float]:
        if not self._jina_key:
            return self._fallback_vector()

        import requests

        headers = {
            "Authorization": f"Bearer {self._jina_key}",
            "Content-Type":  "application/json",
        }
        payload = {"model": _JINA_MODEL, "input": [text]}

        for attempt in range(retries):
            try:
                resp = requests.post(_JINA_URL, headers=headers, json=payload, timeout=70)
                resp.raise_for_status()
                vector: list[float] = resp.json()["data"][0]["embedding"]

                if self.embedding_dim is None:
                    self.embedding_dim = len(vector)
                elif len(vector) != self.embedding_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch: "
                        f"expected {self.embedding_dim}, got {len(vector)}. "
                        "Rebuild the index if the model changed."
                    )
                return vector

            except ValueError:
                raise
            except Exception as exc:
                wait = 2 ** attempt
                log.warning("[RAG] Embed attempt %d/%d failed: %s – retrying in %ds",
                            attempt + 1, retries, exc, wait)
                if attempt < retries - 1:
                    time.sleep(wait)

        log.error("[RAG] All embed attempts failed; returning zero vector.")
        return self._fallback_vector()

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _file_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(65_536), b""):
                h.update(block)
        return h.hexdigest()

    def _read_file(self, path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            return self._read_pdf(path)
        return path.read_text(encoding="utf-8", errors="ignore")

    def _read_pdf(self, path: Path) -> str:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyPDF2 is required: pip install PyPDF2") from exc

        reader = PdfReader(str(path))
        pages  = reader.pages
        content_pages = pages[5:-5] if len(pages) > 20 else pages
        parts: list[str] = []
        for page in content_pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                parts.append(page_text)
        return "\n".join(parts)

    def _chunk_text(self, text: str, max_words: int = 200, overlap: int = 40) -> list[str]:
        words = text.split()
        if not words:
            return []
        step   = max(max_words - overlap, 1)
        chunks: list[str] = []
        for start in range(0, len(words), step):
            chunk = " ".join(words[start: start + max_words]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    # ── FAISS ─────────────────────────────────────────────────────────────────

    def _rebuild_role_chunk_map(self) -> dict[str, list[int]]:
        role_map: dict[str, list[int]] = {}
        for global_idx, chunk in enumerate(self._chunks):
            role_map.setdefault(chunk["role"], []).append(global_idx)
        return role_map

    def _build_faiss(self) -> None:
        with self._lock:
            if self._embeddings is None or len(self._embeddings) == 0:
                log.warning("[RAG] No embeddings – FAISS indexes not built.")
                return

            role_chunk_map = self._rebuild_role_chunk_map()
            new_indexes: dict[str, faiss.Index] = {}

            self.vector_dir.mkdir(parents=True, exist_ok=True)

            for role_key, global_indices in role_chunk_map.items():
                vectors = self._embeddings[global_indices].astype(np.float32).copy()
                
                # Fix 4: Zero-vector normalization guard
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = np.divide(vectors, norms, out=vectors, where=norms > 0)
                
                dim       = vectors.shape[1]
                role_idx  = faiss.IndexFlatIP(dim)
                role_idx.add(vectors)
                new_indexes[role_key] = role_idx
                faiss.write_index(role_idx, str(self._role_index_path(role_key)))
                log.info("[RAG] %s index: %d vectors, dim=%d", role_key, len(global_indices), dim)

            self._role_indexes   = new_indexes
            self._role_chunk_map = role_chunk_map

            # Fix 2: Orphaned index cleanup
            existing_roles = set(role_chunk_map.keys())
            for index_file in self.vector_dir.glob("kb_*.index"):
                role_name = index_file.stem.replace("kb_", "")
                if role_name not in existing_roles:
                    log.info("[RAG] Removing orphaned index file: %s", index_file.name)
                    index_file.unlink(missing_ok=True)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist(self) -> None:
        with self._lock:
            self.vector_dir.mkdir(parents=True, exist_ok=True)
            
            # Fix 6: Atomic writes via tmp files
            tmp_index = self.index_path.with_suffix(".tmp.json")
            tmp_index.write_text(json.dumps(self._chunks, ensure_ascii=False), encoding="utf-8")
            tmp_index.replace(self.index_path)

            if self._embeddings is not None and len(self._embeddings):
                tmp_emb = self.embeddings_path.with_suffix(".tmp.npy")
                np.save(tmp_emb, self._embeddings, allow_pickle=False)
                tmp_emb.replace(self.embeddings_path)

        self._build_faiss()
        log.info("[RAG] Index saved: %d chunks.", len(self._chunks))

    # ── Index loading ─────────────────────────────────────────────────────────

    def _load_index(self) -> None:
        if self.index_path.exists():
            try:
                self._chunks = json.loads(self.index_path.read_text(encoding="utf-8"))
                
                if self.embeddings_path.exists():
                    # Fix 5: Secure deserialization
                    self._embeddings = np.load(self.embeddings_path, allow_pickle=False)
                    if len(self._embeddings):
                        self.embedding_dim = self._embeddings.shape[1]
                else:
                    self._migrate_legacy_embeddings()

                role_chunk_map = self._rebuild_role_chunk_map()
                
                # Fix 1: Load partial FAISS files instead of all-or-nothing
                for role_key in _ROLE_PATTERNS:
                    p = self._role_index_path(role_key)
                    if p.exists():
                        self._role_indexes[role_key]   = faiss.read_index(str(p))
                        self._role_chunk_map[role_key] = role_chunk_map.get(role_key, [])

                log.info("[RAG] Loaded %d chunks from disk.", len(self._chunks))
                self._auto_ingest_pdfs()

                need_rebuild = False
                fresh_map    = self._rebuild_role_chunk_map()
                for role_key, indices in fresh_map.items():
                    stored = self._role_indexes.get(role_key)
                    if stored is None or stored.ntotal != len(indices):
                        log.warning("[RAG] %s index out of sync – rebuilding.", role_key)
                        need_rebuild = True
                        break
                
                if need_rebuild:
                    self._build_faiss()
                return

            except Exception as exc:
                log.error("[RAG] Could not load existing index (%s) – rebuilding.", exc)
                self._chunks         = []
                self._embeddings     = None
                self._role_indexes   = {}
                self._role_chunk_map = {}

        # First run (Cold start)
        self._chunks     = []
        self._embeddings = None
        for kb_file in self.kb_root.glob("*.txt"):
            self._ingest_file(kb_file, role=kb_file.stem, persist=False)
        self._auto_ingest_pdfs(persist=False)
        self._persist()

    def _migrate_legacy_embeddings(self) -> None:
        log.info("[RAG] Migrating legacy inline embeddings to .npy …")
        first_valid_dim = next((len(c["embedding"]) for c in self._chunks if c.get("embedding")), 1024)
        if self.embedding_dim is None:
            self.embedding_dim = first_valid_dim

        vectors: list[list[float]] = []
        clean:   list[dict]        = []
        for chunk in self._chunks:
            emb = chunk.pop("embedding", None)
            vectors.append(emb if emb else [0.0] * self.embedding_dim)
            clean.append(chunk)

        self._chunks     = clean
        self._embeddings = np.array(vectors, dtype=np.float32)

    # ── Auto-ingest ───────────────────────────────────────────────────────────

    def _auto_ingest_pdfs(self, persist: bool = True) -> None:
        stored_hashes: dict[str, str] = {}
        for chunk in self._chunks:
            src = chunk["source"]
            if src not in stored_hashes:
                stored_hashes[src] = chunk.get("metadata", {}).get("file_hash", "")

        added = 0
        for pdf_file in sorted(self.kb_root.glob("*.pdf")):
            current_hash = self._file_hash(pdf_file)
            stored_hash  = stored_hashes.get(pdf_file.name, "")

            if stored_hash == current_hash:
                continue

            if stored_hash:
                log.info("[RAG] PDF changed, re-ingesting: %s", pdf_file.name)
                with self._lock:
                    old_indices = [i for i, c in enumerate(self._chunks) if c["source"] == pdf_file.name]
                    for i in sorted(old_indices, reverse=True):
                        self._chunks.pop(i)
                    if self._embeddings is not None and old_indices:
                        self._embeddings = np.delete(self._embeddings, old_indices, axis=0)
            else:
                log.info("[RAG] New PDF: %s", pdf_file.name)

            role = self._detect_role_from_file(pdf_file)
            self._ingest_file(
                pdf_file, role=role,
                metadata={"file_hash": current_hash},
                persist=False,
            )
            added += 1

        if added and persist:
            self._persist()

    def _detect_role_from_file(self, path: Path) -> str:
        # Fix 8: Semantic check fallback instead of purely naive filename scraping
        base_role = self._role_key(path.name)
        if base_role != "backend_engineer":
            return base_role
        
        try:
            if path.suffix.lower() == ".pdf":
                from PyPDF2 import PdfReader # type: ignore
                reader = PdfReader(str(path))
                sample_text = "".join((page.extract_text() or "") for page in reader.pages[:2])
            else:
                sample_text = path.read_text(encoding="utf-8", errors="ignore")[:1000]
            
            # Reparse the first page/1000 characters to determine domain
            return self._role_key(sample_text)
        except Exception:
            return "backend_engineer"

    # ── Core ingest ───────────────────────────────────────────────────────────

    def _ingest_file(
        self,
        path:     Path,
        role:     str,
        metadata: Optional[Dict] = None,
        persist:  bool           = True,
    ) -> bool:
        if not path.exists():
            log.warning("[RAG] File not found: %s", path)
            return False
        try:
            raw_text = self._read_file(path)
        except Exception as exc:
            log.error("[RAG] Could not read %s: %s", path.name, exc)
            return False

        role_key = self._role_key(role)
        metadata = metadata or {}

        with self._lock:
            existing_ids = {chunk["id"] for chunk in self._chunks}

        new_chunks:  list[dict]        = []
        new_vectors: list[list[float]] = []

        for idx, chunk_text in enumerate(self._chunk_text(raw_text)):
            chunk_id = hashlib.sha256(
                f"{role_key}:{path.name}:{idx}:{chunk_text}".encode()
            ).hexdigest()
            
            if chunk_id in existing_ids:
                continue
            
            try:
                emb = self._embed(chunk_text)
            except ValueError as exc:
                log.error("[RAG] Aborting ingest of %s: %s", path.name, exc)
                return False
                
            new_chunks.append({
                "id":          chunk_id,
                "role":        role_key,
                "source":      path.name,
                "chunk_index": idx,
                "content":     chunk_text,
                "metadata":    metadata,
            })
            new_vectors.append(emb)

        if not new_chunks:
            return True

        with self._lock:
            self._chunks.extend(new_chunks)
            new_np = np.array(new_vectors, dtype=np.float32)
            if self._embeddings is None or len(self._embeddings) == 0:
                self._embeddings = new_np
            else:
                self._embeddings = np.vstack([self._embeddings, new_np])
                
            if self.embedding_dim is None and len(self._embeddings):
                self.embedding_dim = self._embeddings.shape[1]

        if persist:
            self._persist()
        return True

    def ingest_document(
        self,
        file_path: str,
        role:      str,
        metadata:  Optional[Dict] = None,
        persist:   bool           = True,
    ) -> bool:
        return self._ingest_file(Path(file_path), role=role, metadata=metadata, persist=persist)

    # ── Query construction ────────────────────────────────────────────────────

    def construct_query(
        self,
        resume_info:          Dict,
        role:                 str,
        conversation_history: Optional[List] = None,
    ) -> str:
        def clean(lst: list) -> list[str]:
            return [str(s) for s in lst if s]

        skills       = clean(resume_info.get("skills")       or [])
        technologies = clean(resume_info.get("technologies") or [])
        domains      = clean(resume_info.get("domains")      or [])
        experience   = resume_info.get("experience_years")

        parts = [f"{role} technical interview"]
        if skills:
            parts.append("skills " + " ".join(skills[:8]))
        if technologies:
            parts.append("technologies " + " ".join(technologies[:8]))
        if domains:
            parts.append("domains " + " ".join(domains[:5]))
        if experience is not None:
            level = "senior" if experience >= 5 else "intermediate" if experience >= 2 else "junior"
            parts.append(f"{level} applied concepts")
        if conversation_history:
            recent = [
                e.get("topic", "")
                for e in conversation_history[-3:]
                if isinstance(e, dict) and e.get("topic")
            ]
            if recent:
                parts.append("next topic after " + " ".join(recent))
        return " ".join(parts)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve_context(
        self,
        query: str,
        role:  str,
        top_k: int = 5,
    ) -> List[Dict]:
        role_key   = self._role_key(role)
        role_index = self._role_indexes.get(role_key)

        if role_index is None:
            log.warning("[RAG] No FAISS index for role '%s' – attempting build.", role_key)
            self._build_faiss()
            role_index = self._role_indexes.get(role_key)
            if role_index is None:
                log.error("[RAG] Cannot retrieve: no chunks for role '%s'.", role_key)
                return []

        role_map  = self._role_chunk_map.get(role_key, [])
        query_emb = np.array([self._embed(query)], dtype=np.float32)
        
        # Fix 4: Guard zero-vector single queries
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm

        # Fix 7: Clamp request bounds
        search_k = min(top_k, role_index.ntotal)
        if search_k == 0:
            return []
            
        scores, indices = role_index.search(query_emb, search_k)

        results: list[dict] = []
        for score, local_idx in zip(scores[0], indices[0]):
            if local_idx == -1:
                continue
            if local_idx >= len(role_map):
                log.warning("[RAG] local_idx %d out of range for role %s (map size %d)",
                            local_idx, role_key, len(role_map))
                continue
            
            global_idx = role_map[local_idx]
            if global_idx >= len(self._chunks):
                log.warning("[RAG] global_idx %d out of range (chunks: %d)",
                            global_idx, len(self._chunks))
                continue
            
            chunk = self._chunks[global_idx]
            results.append({
                "content":  chunk["content"],
                "metadata": {
                    "role":        chunk["role"],
                    "source":      chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    **chunk.get("metadata", {}),
                },
                "score": float(score),
            })

        return results

    # ── Question generation ───────────────────────────────────────────────────

    def generate_questions(
        self,
        context:       List[Dict],
        resume_info:   Dict,
        role:          str,
        num_questions: int = 3,
    ) -> List[Dict]:
        api_key = self._load_secret("GROQ_API_KEY")
        if api_key:
            return self._generate_with_llm(context, resume_info, role, num_questions, api_key)
        return self._generate_template_questions(context, resume_info, role, num_questions)

    def _generate_with_llm(
        self,
        context:       List[Dict],
        resume_info:   Dict,
        role:          str,
        num_questions: int,
        api_key:       str,
    ) -> List[Dict]:
        try:
            from groq import Groq  # type: ignore
        except ImportError:
            log.warning("[RAG] groq package not installed – template fallback.")
            return self._generate_template_questions(context, resume_info, role, num_questions)

        kb_text    = "\n\n---\n\n".join(
            f"[Source: {c['metadata']['source']}]\n{c['content']}" for c in context[:4]
        )
        skills_str = ", ".join(resume_info.get("skills", [])[:8]) or "not specified"
        exp        = resume_info.get("experience_years") or 0
        seniority  = "senior" if exp >= 5 else "mid-level" if exp >= 2 else "junior"

        prompt = f"""You are a technical interviewer for a {role} position.

CANDIDATE PROFILE
Seniority : {seniority}
Skills    : {skills_str}

RETRIEVED KNOWLEDGE BASE CONTEXT (from ML/AI textbooks)
{kb_text}

TASK
Generate exactly {num_questions} technical interview questions.

Rules:
- Every question MUST be grounded in the knowledge base context above.
- Reference concepts, terminology, or examples found in that context.
- Questions should match the candidate seniority ({seniority}).
- Cover different topics across the questions.
- Do NOT use generic filler questions.

Return ONLY valid JSON with this exact shape:
{{
  "questions": [
    {{
      "question": "<the interview question>",
      "topic": "<short concept label>",
      "difficulty": "<easy|medium|hard>",
      "expected_concepts": ["<concept1>", "<concept2>"],
      "context": "<specific KB text this question draws from (max 200 chars)>"
    }}
  ]
}}"""

        try:
            client   = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=_GROQ_MODEL,
                temperature=0.7,
                max_tokens=1400,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content or ""
            raw = re.sub(r"```json\s*", "", raw)
            raw = re.sub(r"```\s*",     "", raw)

            parsed = json.loads(raw.strip())

            if isinstance(parsed, dict):
                questions = parsed.get("questions") or list(parsed.values())
            elif isinstance(parsed, list):
                questions = parsed
            else:
                raise ValueError("Unexpected JSON structure from LLM.")

            if isinstance(questions, list) and questions:
                return questions[:num_questions]

        except Exception as exc:
            log.warning("[RAG] LLM generation failed (%s) – template fallback.", exc)

        return self._generate_template_questions(context, resume_info, role, num_questions)

    def _generate_template_questions(
        self,
        context:       List[Dict],
        resume_info:   Dict,
        role:          str,
        num_questions: int,
    ) -> List[Dict]:
        focus         = self._candidate_focus(resume_info)
        context_terms = self._context_terms(context)
        source        = context[0]["metadata"]["source"] if context else "knowledge base"
        snippet       = context[0]["content"][:200]      if context else ""
        topic_a       = context_terms[0] if context_terms          else "ML concepts"
        topic_b       = context_terms[1] if len(context_terms) > 1 else "model selection"
        focus_a       = focus[0]         if focus                   else "your chosen algorithm"

        return [
            {
                "question": (
                    f"Based on this concept from {source}: '{snippet}…' — "
                    f"how would you apply this in a {role} context?"
                ),
                "topic":             topic_a,
                "difficulty":        "medium",
                "expected_concepts": context_terms[:3],
                "context":           snippet,
            },
            {
                "question": f"Explain the trade-offs of {topic_b} in a production system.",
                "topic":             topic_b,
                "difficulty":        "medium",
                "expected_concepts": context_terms[:3],
                "context":           snippet,
            },
            {
                "question": (
                    f"How would you evaluate whether {focus_a} "
                    "is the right approach for a given dataset?"
                ),
                "topic":             focus_a,
                "difficulty":        "hard",
                "expected_concepts": focus[:3] + context_terms[:2],
                "context":           snippet,
            },
        ][:num_questions]

    # ── Answer evaluation ─────────────────────────────────────────────────────

    def evaluate_answer(
        self,
        question:          str,
        answer:            str,
        expected_concepts: List[str],
    ) -> Dict:
        if not answer.strip():
            return {
                "score":            0.0,
                "feedback":         "No answer provided.",
                "concepts_covered": [],
                "concepts_missing": expected_concepts,
            }

        answer_tokens   = _concept_tokens(answer)
        question_tokens = _concept_tokens(question)

        covered   = [c for c in expected_concepts if _concept_tokens(c) & answer_tokens]
        missing   = [c for c in expected_concepts if c not in covered]
        relevance = len(answer_tokens & question_tokens) / max(len(question_tokens), 1)
        coverage  = len(covered) / max(len(expected_concepts), 1) if expected_concepts else 1.0
        depth     = min(len(answer.split()) / 70.0, 1.0)
        score     = round((coverage * 55) + (depth * 30) + (relevance * 15), 2)

        parts: list[str] = []
        if coverage >= 0.6:
            parts.append("Strong coverage of the expected concepts")
        else:
            parts.append("Good start, but connect more directly to the expected concepts")
        if depth < 0.45:
            parts.append("add a concrete example, trade-off, or implementation detail")
        if missing:
            parts.append("missing: " + ", ".join(missing[:3]))

        return {
            "score":            score,
            "feedback":         ". ".join(parts) + ".",
            "concepts_covered": covered,
            "concepts_missing": missing,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _candidate_focus(self, resume_info: Dict) -> list[str]:
        focus: list[str] = []
        for field in ("skills", "technologies", "domains"):
            focus.extend(str(s) for s in (resume_info.get(field) or []) if s)
        return list(dict.fromkeys(focus))

    def _context_terms(self, contexts: Iterable[Dict]) -> list[str]:
        stop = {
            "the", "and", "for", "with", "that", "this", "from",
            "into", "are", "how", "what", "also", "such", "can",
        }
        # Fix 9: Python < 3.9 type fallback
        counts: TCounter[str] = Counter()
        for ctx in contexts:
            counts.update(
                t for t in _tokens(ctx["content"])
                if t not in stop and len(t) > 3
            )
        return [term for term, _ in counts.most_common(8)]

    def get_status(self) -> Dict:
        role_counts: dict[str, int] = {}
        sources: set[str] = set()
        with self._lock:
            for chunk in self._chunks:
                role_counts[chunk["role"]] = role_counts.get(chunk["role"], 0) + 1
                sources.add(chunk["source"])
        return {
            "total_chunks":  len(self._chunks),
            "by_role":       role_counts,
            "sources":       sorted(sources),
            "indexes_ready": {k: v.ntotal for k, v in self._role_indexes.items()},
            "embedding_dim": self.embedding_dim,
        }


# ── Lazy singleton ────────────────────────────────────────────────────────────

_pipeline_instance: RAGPipeline | None = None
_pipeline_lock = threading.RLock()

def get_rag_pipeline() -> RAGPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        with _pipeline_lock:
            if _pipeline_instance is None:
                _pipeline_instance = RAGPipeline()
    return _pipeline_instance