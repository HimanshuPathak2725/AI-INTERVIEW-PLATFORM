from __future__ import annotations

import hashlib
import json
import math
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
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


class RAGPipeline:
    """Small local RAG pipeline with persisted hashed embeddings.

    This avoids heavy native vector dependencies while still providing the core
    assignment flow: chunking, embeddings, vector persistence, retrieval, and
    traceable question generation.
    """

    def __init__(self) -> None:
        self.backend_root = Path(__file__).resolve().parents[2]
        self.kb_root = self.backend_root / "knowledge_base"
        self.index_path = self.backend_root / "vector_store" / "kb_index.json"
        self._chunks: list[dict] = []
        self._load_index()

    def _role_key(self, role: str) -> str:
        normalized = role.lower().replace("-", " ").replace("_", " ")
        if "ai" in normalized or "ml" in normalized or "machine learning" in normalized:
            return "ai_ml_engineer"
        if "front" in normalized:
            return "frontend_engineer"
        return "backend_engineer"

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * VECTOR_SIZE
        counts = Counter(_tokens(text))
        if not counts:
            return vector

        for token, count in counts.items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % VECTOR_SIZE
            sign = 1 if digest[2] % 2 == 0 else -1
            vector[index] += sign * (1 + math.log(count))
        return vector

    def _chunk_text(self, text: str, max_words: int = 140, overlap: int = 30) -> list[str]:
        words = text.split()
        if not words:
            return []

        chunks = []
        step = max(max_words - overlap, 1)
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + max_words]).strip()
            if chunk:
                chunks.append(chunk)
            if start + max_words >= len(words):
                break
        return chunks

    def _load_index(self) -> None:
        if self.index_path.exists():
            self._chunks = json.loads(self.index_path.read_text(encoding="utf-8"))
            return

        self._chunks = []
        for kb_file in self.kb_root.glob("*.txt"):
            self.ingest_document(str(kb_file), role=kb_file.stem, persist=False)
        self._persist()

    def _persist(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(self._chunks, indent=2), encoding="utf-8")

    def ingest_document(self, file_path: str, role: str, metadata: Optional[Dict] = None, persist: bool = True) -> bool:
        source = Path(file_path)
        if not source.exists():
            return False

        text = source.read_text(encoding="utf-8", errors="ignore")
        role_key = self._role_key(role)
        metadata = metadata or {}

        existing_ids = {chunk["id"] for chunk in self._chunks}
        for index, chunk in enumerate(self._chunk_text(text)):
            chunk_id = hashlib.sha256(f"{role_key}:{source.name}:{index}:{chunk}".encode("utf-8")).hexdigest()
            if chunk_id in existing_ids:
                continue

            self._chunks.append(
                {
                    "id": chunk_id,
                    "role": role_key,
                    "source": source.name,
                    "chunk_index": index,
                    "content": chunk,
                    "embedding": self._embed(chunk),
                    "metadata": metadata,
                }
            )

        if persist:
            self._persist()
        return True

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
            recent = [entry.get("topic", "") for entry in conversation_history[-3:] if entry.get("topic")]
            if recent:
                parts.append("next topic after " + " ".join(recent))
        return " ".join(parts)

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

    def _difficulty(self, resume_info: Dict, index: int) -> str:
        experience = resume_info.get("experience_years")
        if experience is not None and experience >= 5:
            return "hard" if index else "medium"
        if experience is not None and experience < 2:
            return "easy" if index == 0 else "medium"
        return "medium" if index < 2 else "hard"

    def _candidate_focus(self, resume_info: Dict) -> list[str]:
        focus = []
        for field in ("skills", "technologies", "domains"):
            focus.extend(resume_info.get(field, []) or [])
        return list(dict.fromkeys(focus))

    def _context_terms(self, contexts: Iterable[Dict]) -> list[str]:
        stop = {"the", "and", "for", "with", "that", "this", "from", "into", "are", "how", "what"}
        counts = Counter()
        for context in contexts:
            counts.update(token for token in _tokens(context["content"]) if token not in stop and len(token) > 3)
        return [term for term, _ in counts.most_common(8)]

    def generate_questions(
        self,
        context: List[Dict],
        resume_info: Dict,
        role: str,
        num_questions: int = 3,
    ) -> List[Dict]:
        focus = self._candidate_focus(resume_info)
        context_terms = self._context_terms(context)
        source_label = context[0]["metadata"]["source"] if context else "role knowledge base"
        role_name = role.replace("_", " ")
        article = "an" if role_name[:1].lower() in {"a", "e", "i", "o", "u"} else "a"

        templates = [
            "Using the retrieved material from {source}, explain how {topic} applies to {article} {role} problem you might see in production.",
            "Design a solution involving {topic}. What tradeoffs, failure modes, and validation steps would you discuss?",
            "Compare two approaches for {topic} and describe when you would choose each in a real interview scenario.",
            "Given a candidate background in {topic}, ask a follow-up that tests conceptual depth and practical implementation.",
            "How would you evaluate whether a solution based on {topic} is correct, scalable, and maintainable?",
        ]

        questions = []
        topics = focus + context_terms + ["system design", "retrieval quality", "data modeling"]
        for index in range(num_questions):
            topic = topics[index % len(topics)]
            template = templates[index % len(templates)]
            expected = list(dict.fromkeys([topic, *context_terms[:2], "tradeoffs", "examples"]))[:5]
            questions.append(
                {
                    "question": template.format(source=source_label, topic=topic, role=role_name, article=article),
                    "topic": topic,
                    "difficulty": self._difficulty(resume_info, index),
                    "expected_concepts": expected,
                    "context": context[index % len(context)]["content"] if context else "",
                }
            )
        return questions

    def evaluate_answer(self, question: str, answer: str, expected_concepts: List[str]) -> Dict:
        if not answer.strip():
            return {"score": 0.0, "feedback": "No answer provided.", "concepts_covered": [], "concepts_missing": expected_concepts}

        answer_tokens = _concept_tokens(answer)
        question_tokens = _concept_tokens(question)
        covered = [
            concept
            for concept in expected_concepts
            if _concept_tokens(concept) & answer_tokens
        ]
        missing = [concept for concept in expected_concepts if concept not in covered]
        relevance = len(answer_tokens & question_tokens) / max(len(question_tokens), 1)
        coverage = len(covered) / max(len(expected_concepts), 1)
        depth = min(len(answer.split()) / 70.0, 1.0)
        score = round((coverage * 55) + (depth * 30) + (relevance * 15), 2)

        feedback_parts = []
        if coverage >= 0.6:
            feedback_parts.append("Strong coverage of the expected concepts")
        else:
            feedback_parts.append("Good start, but the answer should connect more directly to the expected concepts")
        if depth < 0.45:
            feedback_parts.append("add a concrete example, tradeoff, or implementation detail")
        if missing:
            feedback_parts.append("missing: " + ", ".join(missing[:3]))

        return {
            "score": score,
            "feedback": ". ".join(feedback_parts) + ".",
            "concepts_covered": covered,
            "concepts_missing": missing,
        }


rag_pipeline = RAGPipeline()
