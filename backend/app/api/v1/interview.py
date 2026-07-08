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

        # Extract content cleanly. If it's a list/dict, handle it safely on backend before shipping
        raw_content = final_state["messages"][-1].content
        if isinstance(raw_content, list):
            extracted_text = ""
            for block in raw_content:
                if isinstance(block, dict) and "text" in block:
                    extracted_text += block["text"]
                elif hasattr(block, "text"):
                    extracted_text += block.text
                else:
                    extracted_text += str(block)
            latest_response = extracted_text
        elif isinstance(raw_content, dict):
            latest_response = raw_content.get("text", str(raw_content))
        else:
            latest_response = str(raw_content)

        # Directly send a flat clean string for response to keep Pydantic safe next round
        return {
            "response": latest_response,
            "current_phase": final_state.get("current_phase", payload.current_phase),
            "evaluation": final_state.get("evaluation_scores", {"technical_accuracy": 0.0})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph Execution Error: {str(e)}")
