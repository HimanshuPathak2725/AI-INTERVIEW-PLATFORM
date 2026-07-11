import asyncio
from pathlib import Path
from backend.app.services.rag_pipeline import get_rag_pipeline

async def verify():
    print("🚀 Initializing RAG Pipeline...")
    pipeline = get_rag_pipeline()

    # Wait a couple of seconds to allow background loading task to setup
    await asyncio.sleep(2)

    # Create a dummy knowledge base document if it doesn't exist
    kb_dir = Path("backend/knowledge_base")
    kb_dir.mkdir(parents=True, exist_ok=True)
    test_file = kb_dir / "test_backend_doc.txt"

    test_file.write_text(
        "Python FastAPI backend engineering focuses on clean async design, "
        "database connection pooling, Pydantic data validation, and building scalable API routing.",
        encoding="utf-8"
    )
    print(f"📝 Created dummy test file at: {test_file}")

    print("⚡ Testing Async Ingestion...")
    # Use wait=True to ensure ingestion completes before evaluation
    success = await pipeline.ingest_document_async(str(test_file), role="backend_engineer", wait=True)

    if success:
        print("✅ Success! Document ingested asynchronously.")
        status = pipeline.evaluate_answer(
            question="What is FastAPI backend engineering?",
            answer="It uses async design and Pydantic validation.",
            expected_concepts=["async design", "Pydantic validation"]
        )
        print(f"📊 Evaluator Test Score: {status['score']}/100")
    else:
        print("❌ Ingestion Failed. Check logs or API Key configuration.")

if __name__ == "__main__":
    asyncio.run(verify())
