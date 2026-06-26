from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from app.core.config import settings

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.]{1,}")
VECTOR_SIZE = 384


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _concept_tokens(text: str) -> set[str]:
    return {token[:-1] if token.endswith("s") and len(token) > 4 else token for token in _tokens(text)}


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(v * v for v in left))
    right_norm = math.sqrt(sum(v * v for v in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


class RAGPipeline:
    """RAG pipeline that ingests PDF knowledge-base books and generates
    LLM-grounded interview questions via the Anthropic API."""

    def __init__(self) -> None:
        self.backend_root = Path(__file__).resolve().parents[2]
        self.kb_root = self.backend_root / "knowledge_base"
        self.index_path = self.backend_root / "vector_store" / "kb_index.json"
        self._chunks: list[dict] = []
        self._load_index()

    # ── Role normalisation ────────────────────────────────────────────────────

    def _role_key(self, role: str) -> str:
        n = role.lower().replace("-", " ").replace("_", " ")
        if "ai" in n or "ml" in n or "machine learning" in n or "data" in n:
            return "ai_ml_engineer"
        if "front" in n:
            return "frontend_engineer"
        return "backend_engineer"

    # ── Lightweight hash embedding (no external deps) ────────────────────────

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * VECTOR_SIZE
        counts = Counter(_tokens(text))
        if not counts:
            return vector
        for token, count in counts.items():
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:2], "big") % VECTOR_SIZE
            sign = 1 if digest[2] % 2 == 0 else -1
            vector[index] += sign * (1 + math.log(count))
        return vector

    # ── Text extraction (PDF + TXT) ───────────────────────────────────────────

    def _read_file(self, path: Path) -> str:
        """Read a .txt or .pdf file and return its text content."""
        if path.suffix.lower() == ".pdf":
            return self._read_pdf(path)
        return path.read_text(encoding="utf-8", errors="ignore")

    def _read_pdf(self, path: Path) -> str:
        """Extract text from a PDF using PyPDF2."""
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise RuntimeError("PyPDF2 is required: pip install PyPDF2")

        text_parts: list[str] = []
        reader = PdfReader(str(path))
        # Skip the first 5 pages (usually TOC/cover) and last 5 (index)
        pages = reader.pages
        content_pages = pages[5:-5] if len(pages) > 10 else pages
        for page in content_pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)
        return "\n".join(text_parts)

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, max_words: int = 200, overlap: int = 40) -> list[str]:
        """Split text into overlapping word-level chunks."""
        words = text.split()
        if not words:
            return []
        chunks: list[str] = []
        step = max(max_words - overlap, 1)
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + max_words]).strip()
            if chunk:
                chunks.append(chunk)
            if start + max_words >= len(words):
                break
        return chunks

    # ── Index persistence ─────────────────────────────────────────────────────

    def _load_index(self) -> None:
        if self.index_path.exists():
            self._chunks = json.loads(self.index_path.read_text(encoding="utf-8"))
            # Auto-ingest any PDFs not yet in the index
            self._auto_ingest_pdfs()
            return

        self._chunks = []
        # Ingest .txt files (legacy)
        for kb_file in self.kb_root.glob("*.txt"):
            self._ingest_file(kb_file, role=kb_file.stem, persist=False)
        # Ingest PDF books
        self._auto_ingest_pdfs(persist=False)
        self._persist()

    def _auto_ingest_pdfs(self, persist: bool = True) -> None:
        """Scan knowledge_base/ for PDFs and ingest any not yet indexed."""
        ingested_sources = {chunk["source"] for chunk in self._chunks}
        added = 0
        for pdf_file in sorted(self.kb_root.glob("*.pdf")):
            if pdf_file.name in ingested_sources:
                continue
            role = self._detect_role_from_filename(pdf_file.name)
            print(f"[RAG] Ingesting PDF: {pdf_file.name} → role={role}")
            self._ingest_file(pdf_file, role=role, persist=False)
            added += 1
        if added and persist:
            self._persist()

    def _detect_role_from_filename(self, filename: str) -> str:
        lower = filename.lower()
        ml_keywords = ["mitchell", "burkov", "hundred", "beginners", "python", "brownlee",
                       "algorithms", "bishop", "pattern", "deep_learning", "machine"]
        for kw in ml_keywords:
            if kw in lower:
                return "ai_ml_engineer"
        return "ai_ml_engineer"  # default for this assignment

    def _ingest_file(self, path: Path, role: str, metadata: Optional[Dict] = None, persist: bool = True) -> bool:
        if not path.exists():
            return False
        try:
            text = self._read_file(path)
        except Exception as exc:
            print(f"[RAG] Could not read {path.name}: {exc}")
            return False

        role_key = self._role_key(role)
        metadata = metadata or {}
        existing_ids = {chunk["id"] for chunk in self._chunks}

        for index, chunk in enumerate(self._chunk_text(text)):
            chunk_id = hashlib.sha256(
                f"{role_key}:{path.name}:{index}:{chunk}".encode()
            ).hexdigest()
            if chunk_id in existing_ids:
                continue
            self._chunks.append({
                "id": chunk_id,
                "role": role_key,
                "source": path.name,
                "chunk_index": index,
                "content": chunk,
                "embedding": self._embed(chunk),
                "metadata": metadata,
            })

        if persist:
            self._persist()
        return True

    # Keep backward-compat alias used by ingest_kb.py
    def ingest_document(self, file_path: str, role: str, metadata: Optional[Dict] = None, persist: bool = True) -> bool:
        return self._ingest_file(Path(file_path), role=role, metadata=metadata, persist=persist)

    def _persist(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(self._chunks, indent=2), encoding="utf-8")
        print(f"[RAG] Index saved: {len(self._chunks)} chunks")

    # ── Query construction ────────────────────────────────────────────────────

    def construct_query(
        self,
        resume_info: Dict,
        role: str,
        conversation_history: Optional[List] = None,
    ) -> str:
        skills = resume_info.get("skills", []) or []
        technologies = resume_info.get("technologies", []) or []
        domains = resume_info.get("domains", []) or []
        experience = resume_info.get("experience_years")

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
            recent = [e.get("topic", "") for e in conversation_history[-3:] if e.get("topic")]
            if recent:
                parts.append("next topic after " + " ".join(recent))
        return " ".join(parts)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve_context(self, query: str, role: str, top_k: int = 5) -> List[Dict]:
        if not self._chunks:
            self._load_index()

        query_vector = self._embed(query)
        role_key = self._role_key(role)
        scored = []
        for chunk in self._chunks:
            if chunk["role"] != role_key:
                continue
            score = _cosine(query_vector, chunk["embedding"])
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "content": chunk["content"],
                "metadata": {
                    "role": chunk["role"],
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    **chunk.get("metadata", {}),
                },
                "score": round(score, 4),
            }
            for score, chunk in scored[:top_k]
        ]

    # ── LLM-based question generation ─────────────────────────────────────────

    def generate_questions(
        self,
        context: List[Dict],
        resume_info: Dict,
        role: str,
        num_questions: int = 3,
    ) -> List[Dict]:
        """Generate interview questions grounded in the retrieved KB context.

        Uses the Anthropic API (ANTHROPIC_API_KEY env var) when available.
        Falls back to improved templates if the key is not set.
        """
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if api_key:
            return self._generate_with_llm(context, resume_info, role, num_questions, api_key)
        return self._generate_template_questions(context, resume_info, role, num_questions)

    def _generate_with_llm(
        self,
        context: List[Dict],
        resume_info: Dict,
        role: str,
        num_questions: int,
        api_key: str,
    ) -> List[Dict]:
        """Call Claude to produce context-grounded questions."""
        try:
            import anthropic
        except ImportError:
            print("[RAG] anthropic package not installed, using template fallback")
            return self._generate_template_questions(context, resume_info, role, num_questions)

        kb_text = "\n\n---\n\n".join(
            f"[Source: {c['metadata']['source']}]\n{c['content']}"
            for c in context[:4]
        )
        skills_str = ", ".join(resume_info.get("skills", [])[:8]) or "not specified"
        exp = resume_info.get("experience_years", 0)
        seniority = "senior" if exp >= 5 else "mid-level" if exp >= 2 else "junior"

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
- Questions should match the candidate's seniority ({seniority}).
- Cover different topics across the questions.
- Do NOT use generic filler questions.

Return ONLY valid JSON — a list of {num_questions} objects, no prose, no markdown:
[
  {{
    "question": "<the interview question>",
    "topic": "<short concept label, e.g. 'Gradient Descent'>",
    "difficulty": "<easy|medium|hard>",
    "expected_concepts": ["<concept1>", "<concept2>"],
    "context": "<the specific KB text this question is drawn from (max 200 chars)>"
  }}
]"""

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*", "", raw)

        try:
            questions = json.loads(raw.strip())
            if isinstance(questions, list) and questions:
                return questions
        except (json.JSONDecodeError, KeyError):
            pass

        print("[RAG] LLM response could not be parsed, using template fallback")
        return self._generate_template_questions(context, resume_info, role, num_questions)

    def _generate_template_questions(
        self,
        context: List[Dict],
        resume_info: Dict,
        role: str,
        num_questions: int,
    ) -> List[Dict]:
        """Improved template fallback that at least injects KB content."""
        focus = self._candidate_focus(resume_info)
        context_terms = self._context_terms(context)
        source = context[0]["metadata"]["source"] if context else "knowledge base"
        snippet = context[0]["content"][:200] if context else ""

        base_questions = [
            {
                "question": f"Based on this concept from {source}: '{snippet}...' — how would you apply this in a {role} context?",
                "topic": context_terms[0] if context_terms else "ML concepts",
                "difficulty": "medium",
                "expected_concepts": context_terms[:3],
                "context": snippet,
            },
            {
                "question": f"Explain the trade-offs involved in {context_terms[1] if len(context_terms) > 1 else 'model selection'} for a production system.",
                "topic": context_terms[1] if len(context_terms) > 1 else "model selection",
                "difficulty": "medium",
                "expected_concepts": context_terms[:3],
                "context": snippet,
            },
            {
                "question": f"How would you evaluate whether {focus[0] if focus else 'your chosen algorithm'} is the right approach for a given dataset?",
                "topic": focus[0] if focus else "algorithm evaluation",
                "difficulty": "hard",
                "expected_concepts": focus[:3] + context_terms[:2],
                "context": snippet,
            },
        ]
        return base_questions[:num_questions]

    # ── Answer evaluation ─────────────────────────────────────────────────────

    def evaluate_answer(self, question: str, answer: str, expected_concepts: List[str]) -> Dict:
        if not answer.strip():
            return {"score": 0.0, "feedback": "No answer provided.", "concepts_covered": [], "concepts_missing": expected_concepts}

        answer_tokens = _concept_tokens(answer)
        question_tokens = _concept_tokens(question)
        covered = [c for c in expected_concepts if _concept_tokens(c) & answer_tokens]
        missing = [c for c in expected_concepts if c not in covered]
        relevance = len(answer_tokens & question_tokens) / max(len(question_tokens), 1)
        coverage = len(covered) / max(len(expected_concepts), 1)
        depth = min(len(answer.split()) / 70.0, 1.0)
        score = round((coverage * 55) + (depth * 30) + (relevance * 15), 2)

        parts = []
        if coverage >= 0.6:
            parts.append("Strong coverage of the expected concepts")
        else:
            parts.append("Good start, but connect more directly to the expected concepts")
        if depth < 0.45:
            parts.append("add a concrete example, tradeoff, or implementation detail")
        if missing:
            parts.append("missing: " + ", ".join(missing[:3]))

        return {
            "score": score,
            "feedback": ". ".join(parts) + ".",
            "concepts_covered": covered,
            "concepts_missing": missing,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _candidate_focus(self, resume_info: Dict) -> list[str]:
        focus = []
        for field in ("skills", "technologies", "domains"):
            focus.extend(resume_info.get(field, []) or [])
        return list(dict.fromkeys(focus))

    def _context_terms(self, contexts: Iterable[Dict]) -> list[str]:
        stop = {"the", "and", "for", "with", "that", "this", "from", "into", "are", "how", "what"}
        counts: Counter = Counter()
        for ctx in contexts:
            counts.update(t for t in _tokens(ctx["content"]) if t not in stop and len(t) > 3)
        return [term for term, _ in counts.most_common(8)]

    def get_status(self) -> Dict:
        role_counts: Dict[str, int] = {}
        sources: set = set()
        for chunk in self._chunks:
            role_counts[chunk["role"]] = role_counts.get(chunk["role"], 0) + 1
            sources.add(chunk["source"])
        return {
            "total_chunks": len(self._chunks),
            "by_role": role_counts,
            "sources": sorted(sources),
        }


rag_pipeline = RAGPipeline()