from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.app.agents.state import InterviewState
from backend.app.core.config import settings

# Updated to the latest stable model from the Gemini 3.5 family
llm = ChatGoogleGenerativeAI(
    google_api_key=getattr(settings, "GEMINI_API_KEY", getattr(settings, "OPENAI_API_KEY", "")), 
    model="gemini-3.1-flash-lite",
    temperature=0.7
)

async def retrieval_node(state: InterviewState) -> Dict[str, Any]:
    last_message = state["messages"][-1].content if state["messages"] else ""
    mock_retrieved_context = f"Vetted evaluation criteria and standards related to context: '{last_message[:30]}...'"
    return {"retrieved_context": mock_retrieved_context}

async def evaluation_node(state: InterviewState) -> Dict[str, Any]:
    current_count = state.get("question_count", 0) + 1
    updated_scores = state.get("evaluation_scores", {}).copy() if state.get("evaluation_scores") else {"technical_accuracy": 0.0}
    current_accuracy = updated_scores.get("technical_accuracy", 0.0)
    updated_scores["technical_accuracy"] = min(current_accuracy + 0.1, 1.0)
    return {
        "question_count": current_count,
        "evaluation_scores": updated_scores
    }

async def interviewer_node(state: InterviewState) -> Dict[str, Any]:
    context = state.get("retrieved_context", "")
    phase = state.get("current_phase", "warmup")
    system_prompt = (
        f"You are a Senior Technical Interviewer running the '{phase}' phase of the technical round.\n"
        f"Context from Knowledge Base:\n{context}\n\n"
        "Generate the next concise, clear response or a structured follow-up question."
    )
    messages_payload = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = await llm.ainvoke(messages_payload)
    return {"messages": [AIMessage(content=response.content)]}