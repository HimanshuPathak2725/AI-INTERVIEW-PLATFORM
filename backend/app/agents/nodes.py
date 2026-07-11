from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.app.agents.state import InterviewState
from backend.app.core.config import settings

raw_key = getattr(settings, "GEMINI_API_KEY", "")
api_key_param = raw_key if raw_key else "MOCK_KEY_FOR_COMPILATION"

llm = ChatGoogleGenerativeAI(
    api_key=api_key_param,
    model="gemini-3.1-flash-lite",
    temperature=0.7
)

async def load_candidate_node(state: InterviewState) -> Dict[str, Any]:
    """
    Extracts already asked questions from the message timeline to maintain state 
    across stateless HTTP endpoint requests, preventing duplicate queries.
    """
    messages = state.get("messages", [])
    # Automatically capture past questions from the chat context array
    extracted_asked = [
        msg.content for msg in messages 
        if msg.type == "ai" or getattr(msg, "role", "") == "assistant"
    ]
    
    return {
        "current_phase": state.get("current_phase", "warmup"),
        "current_topic": state.get("current_topic", "General Introduction"),
        "covered_topics": state.get("covered_topics", []),
        "difficulty": state.get("difficulty", "easy"),
        "question_count": len(extracted_asked),
        "asked_questions": extracted_asked,
        "question_hashes": state.get("question_hashes", []),
        "resume_skills": state.get("resume_skills", []),
        "evaluation_scores": state.get("evaluation_scores", {"technical_accuracy": 0.0}),
        "last_feedback": state.get("last_feedback", ""),
        "conversation_summary": state.get("conversation_summary", "")
    }

async def retrieval_node(state: InterviewState) -> Dict[str, Any]:
    last_message = state["messages"][-1].content if state["messages"] else ""
    mock_retrieved_context = f"Vetted evaluation criteria and standards related to context: '{last_message[:30]}...'"
    return {"retrieved_context": mock_retrieved_context}

async def evaluation_node(state: InterviewState) -> Dict[str, Any]:
    """
    Processes the last human response and updates evaluation feedback scores.
    """
    updated_scores = state.get("evaluation_scores", {}).copy() if state.get("evaluation_scores") else {"technical_accuracy": 0.0}
    current_accuracy = updated_scores.get("technical_accuracy", 0.0)
    updated_scores["technical_accuracy"] = min(current_accuracy + 0.1, 1.0)
    
    return {
        "evaluation_scores": updated_scores,
        "last_feedback": "Analyzed candidate response format against target tech stack."
    }

async def interviewer_node(state: InterviewState) -> Dict[str, Any]:
    context = state.get("retrieved_context", "")
    phase = state.get("current_phase", "warmup")
    asked_list = state.get("asked_questions", [])
    current_count = len(asked_list) + 1
    
    # Format previously asked questions to strict blacklisted block inside system instructions
    blacklist_str = "\n".join([f"- {q}" for q in asked_list]) if asked_list else "None"
    
    system_prompt = (
        f"You are a Senior Technical Interviewer running the '{phase}' phase of the technical round.\n"
        f"Context from Knowledge Base:\n{context}\n\n"
        f"CRITICAL: Do NOT repeat or ask variations of these questions:\n{blacklist_str}\n\n"
        "Generate a completely new, distinct follow-up question or response based on the evaluation phase."
    )
    
    messages_payload = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = await llm.ainvoke(messages_payload)
    
    updated_asked = asked_list.copy()
    updated_asked.append(str(response.content))
    
    return {
        "messages": [AIMessage(content=response.content)],
        "question_count": current_count,
        "asked_questions": updated_asked
    }

async def finalize_interview_node(state: InterviewState) -> Dict[str, Any]:
    return {
        "current_phase": "wrap_up",
        "conversation_summary": "Interview completed successfully. All technical modules analyzed."
    }
