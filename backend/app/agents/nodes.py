from typing import Any, Dict, List
import json
import re
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.app.agents.state import InterviewState
from backend.app.core.config import settings

raw_key = getattr(settings, "GEMINI_API_KEY", "")
api_key_param = raw_key if raw_key else "MOCK_KEY_FOR_COMPILATION"

llm = ChatGoogleGenerativeAI(
    api_key=api_key_param,
    model="gemini-3.1-flash-lite",
    temperature=0.3  # Temperature lowered for more consistent evaluation metrics
)


def _count_assistant_messages(messages: List[Any]) -> int:
    return sum(
        1
        for message in messages
        if getattr(message, "type", "") == "ai"
        or getattr(message, "role", "") == "assistant"
        or isinstance(message, AIMessage)
    )


def _format_transcript(messages: List[Any]) -> str:
    transcript_lines: List[str] = []

    for message in messages:
        role = getattr(message, "type", "message")
        if role == "human":
            role = "candidate"
        elif role == "ai":
            role = "interviewer"
        else:
            role = getattr(message, "role", role)

        content = getattr(message, "content", str(message))
        transcript_lines.append(f"- **{role.title()}**: {content}")

    return "\n".join(transcript_lines)

async def load_candidate_node(state: InterviewState) -> Dict[str, Any]:
    """
    Extracts already asked questions from the message timeline to maintain state 
    across stateless HTTP endpoint requests, preventing duplicate queries.
    """
    messages = state.get("messages", [])
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
    Dynamically processes the last candidate response against the interviewer's question 
    using Gemini LLM to generate real-time technical accuracy scores.
    """
    messages = state.get("messages", [])
    updated_scores = {"technical_accuracy": 0.70}  # Smart default fallback
    feedback = "Analyzed candidate response architecture."

    # We need at least the AI question and the User response to evaluate
    if len(messages) >= 2:
        user_answer = messages[-1].content
        ai_question = messages[-2].content

        eval_prompt = (
            "You are an elite Technical Interviewer and System Architect. Your job is to strictly evaluate the candidate's answer against the question asked.\n\n"
            f"Interviewer Question:\n{ai_question}\n\n"
            f"Candidate Answer:\n{user_answer}\n\n"
            "Analyze the accuracy, depth, and edge cases addressed in the candidate's response. "
            "Provide your assessment in exactly this JSON format:\n"
            "{\n"
            "  \"technical_accuracy\": <float between 0.0 and 1.0>,\n"
            "  \"feedback\": \"<brief 1-sentence feedback explaining the score>\"\n"
            "}\n"
            "CRITICAL: Return ONLY valid JSON. Do not write markdown wrapping, no ```json, just raw text."
        )

        try:
            response = await llm.ainvoke([{"role": "user", "content": eval_prompt}])
            clean_content = response.content.strip()
            
            # Clean up potential LLM markdown leaks safely
            clean_content = re.sub(r"^```json\s*|\s*```$", "", clean_content, flags=re.MULTILINE).strip()
            
            data = json.loads(clean_content)
            score = float(data.get("technical_accuracy", 0.70))
            
            # Bound safety limits
            updated_scores["technical_accuracy"] = max(0.0, min(1.0, score))
            feedback = data.get("feedback", "Evaluation processed successfully.")
        except Exception as e:
            print(f"⚠️ Graph LLM Evaluation Parsing Error: {str(e)}")
            # In case of any transient parsing error, keep a reasonable default based on content presence
            updated_scores["technical_accuracy"] = 0.75 if len(user_answer) > 50 else 0.40

    return {
        "evaluation_scores": updated_scores,
        "last_feedback": feedback
    }

async def interviewer_node(state: InterviewState) -> Dict[str, Any]:
    context = state.get("retrieved_context", "")
    phase = state.get("current_phase", "warmup")
    asked_list = state.get("asked_questions", [])
    candidate_profile = state.get("candidate_profile", {})
    resume_skills = state.get("resume_skills", [])
    current_question_count = _count_assistant_messages(state.get("messages", [])) + 1

    if current_question_count >= 5:
        next_phase = "wrap_up"
    elif current_question_count < 2:
        next_phase = "warmup"
    elif current_question_count in (2, 3):
        next_phase = "deep_dive"
    else:
        next_phase = "system_design"

    blacklist_str = "\n".join([f"- {q}" for q in asked_list]) if asked_list else "None"

    system_prompt = (
        f"You are a Senior Technical Interviewer running the '{phase}' phase of the technical round.\n"
        "Your only job is to generate the next sharp technical question.\n\n"
        f"Candidate Profile:\n{json.dumps(candidate_profile, default=str, indent=2)}\n\n"
        f"Resume Skills:\n{json.dumps(resume_skills, default=str)}\n\n"
        f"Knowledge Base Context:\n{context}\n\n"
        f"CRITICAL: Do NOT repeat or ask variations of these questions:\n{blacklist_str}\n\n"
        "Ask exactly one distinct question that probes the candidate's real depth."
    )
    
    messages_payload = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = await llm.ainvoke(messages_payload)
    
    updated_asked = asked_list.copy()
    updated_asked.append(str(response.content))
    
    return {
        "messages": [AIMessage(content=response.content)],
        "question_count": current_question_count,
        "current_phase": next_phase,
        "asked_questions": updated_asked
    }

async def finalize_interview_node(state: InterviewState) -> Dict[str, Any]:
    transcript = _format_transcript(state.get("messages", []))
    candidate_profile = json.dumps(state.get("candidate_profile", {}), default=str, indent=2)
    resume_skills = json.dumps(state.get("resume_skills", []), default=str)
    evaluation_scores = json.dumps(state.get("evaluation_scores", {}), default=str, indent=2)

    summary_prompt = (
        "You are an expert interview evaluator.\n"
        "Produce a concise but complete markdown scorecard for the interview.\n\n"
        "Use the transcript, candidate profile, resume skills, and evaluation metrics below.\n"
        "Return valid markdown only with these sections: Executive Summary, Technical Assessment, Strengths, Gaps, and Final Recommendation.\n\n"
        f"Candidate Profile:\n{candidate_profile}\n\n"
        f"Resume Skills:\n{resume_skills}\n\n"
        f"Evaluation Scores:\n{evaluation_scores}\n\n"
        f"Transcript:\n{transcript}\n"
    )

    response = await llm.ainvoke([
        {"role": "system", "content": summary_prompt},
        {"role": "user", "content": "Generate the final interview evaluation report now."},
    ])

    summary_markdown = str(response.content).strip()

    return {
        "current_phase": "wrap_up",
        "messages": [AIMessage(content=summary_markdown)],
        "conversation_summary": summary_markdown,
    }
