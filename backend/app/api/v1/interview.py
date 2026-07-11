from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import shutil
import os
from langchain_core.messages import HumanMessage, AIMessage
from backend.app.agents.graph import interview_graph

router = APIRouter()

class ChatMessage(BaseModel):
    role: str
    content: str

class InterviewRequest(BaseModel):
    user_input: str
    chat_history: List[ChatMessage]
    current_phase: str = "warmup"


def _serialize_message(message: Any) -> Dict[str, Any]:
    role = getattr(message, "type", "message")
    if role == "ai":
        role = "assistant"
    elif role == "human":
        role = "user"
    else:
        role = getattr(message, "role", role)

    content = getattr(message, "content", message)
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(str(block["text"]))
            elif hasattr(block, "text"):
                text_parts.append(str(block.text))
            else:
                text_parts.append(str(block))
        content = "".join(text_parts)
    elif isinstance(content, dict):
        content = content.get("text", str(content))

    return {"role": role, "content": str(content)}

@router.post("/resume/upload")
async def upload_resume(file: UploadFile = File(...)):
    if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    
    try:
        upload_dir = "backend/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {
            "status": "success",
            "filename": file.filename,
            "message": "Great! Resume successfully uploaded and synchronized with your profile."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing error: {str(e)}")

@router.post("/session/respond")
async def process_interview_turn(payload: InterviewRequest):
    try:
        formatted_messages = []
        for msg in payload.chat_history:
            if msg.role == "user":
                formatted_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                formatted_messages.append(AIMessage(content=msg.content))
        
        formatted_messages.append(HumanMessage(content=payload.user_input))

        initial_state = {
            "messages": formatted_messages,
            "current_phase": payload.current_phase,
            "question_count": len(payload.chat_history) // 2,
            "max_questions_per_phase": 5,
            "evaluation_scores": {"technical_accuracy": 0.0},
            "candidate_profile": {},
            "retrieved_context": ""
        }

        final_state = await interview_graph.ainvoke(initial_state)

        serialized_messages = [_serialize_message(message) for message in final_state.get("messages", [])]
        latest_message = serialized_messages[-1]["content"] if serialized_messages else ""

        # Extract and safely normalize evaluation scores to float (0.0 to 1.0)
        eval_scores = final_state.get("evaluation_scores", {"technical_accuracy": 0.0})
        tech_accuracy_val = eval_scores.get("technical_accuracy", 0.0)

        # Ensure we return a float between 0.0 and 1.0
        # Convert percentage-style values (> 1.0) to decimal
        if tech_accuracy_val > 1.0:
            normalized_score = float(tech_accuracy_val / 100.0)
        else:
            normalized_score = float(tech_accuracy_val)

        # Clamp to valid range [0.0, 1.0]
        normalized_score = max(0.0, min(1.0, normalized_score))

        summary_markdown = (
            final_state.get("conversation_summary")
            or (latest_message if final_state.get("current_phase") == "wrap_up" else "")
        )

        return {
            "response": summary_markdown or latest_message,
            "current_phase": final_state.get("current_phase", payload.current_phase),
            "technical_accuracy": normalized_score,
            "summary": summary_markdown,
            "messages": serialized_messages,
            "evaluation": {"technical_accuracy": normalized_score},
            "evaluation_scores": {"technical_accuracy": normalized_score}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph Execution Error: {str(e)}")
