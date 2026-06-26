#!/usr/bin/env python3
"""Ingest role-specific knowledge base documents into the local vector index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag_pipeline import rag_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a knowledge base text file")
    parser.add_argument("--role", required=True, help="Role identifier")
    parser.add_argument("--file", required=True, help="Path to the document file")
    parser.add_argument("--metadata", default="", help="Optional metadata")
    args = parser.parse_args()

    source = Path(args.file)
    if not source.exists():
        print(f"Error: File not found: {source}")
        return 1

    metadata = {}
    if args.metadata:
        for item in args.metadata.split(","):
            if ":" in item:
                key, value = item.split(":", 1)
                metadata[key.strip()] = value.strip()

    success = rag_pipeline.ingest_document(str(source), role=args.role, metadata=metadata)
    if not success:
        print("Knowledge base ingestion failed.")
        return 1

    text = source.read_text(encoding="utf-8", errors="ignore")
    print(f"Ingesting document for role: {args.role}")
    print(f"File: {source}")
    print(f"Lines: {len(text.splitlines())}")
    print(f"Index: {rag_pipeline.index_path}")
    print("Knowledge base ingestion completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
