from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from backend.app.agents.graph import interview_graph

router = APIRouter()

class InterviewRequest(BaseModel):
    user_input: str
    chat_history: List[Dict[str, str]]
    current_phase: str = "warmup"

@router.post("/session/respond")
async def process_interview_turn(payload: InterviewRequest):
    try:
        # 1. Reconstruct baseline message structures from incoming JSON history
        formatted_messages = []
        for msg in payload.chat_history:
            if msg.get("role") == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                formatted_messages.append(AIMessage(content=msg["content"]))
        
        # Add the latest user input to the execution frame
        formatted_messages.append(HumanMessage(content=payload.user_input))

        # 2. Structure initial dict state values for LangGraph invocation
        initial_state = {
            "messages": formatted_messages,
            "current_phase": payload.current_phase,
            "question_count": len(payload.chat_history) // 2,
            "max_questions_per_phase": 5,
            "evaluation_scores": {"technical_accuracy": 0.0},
            "candidate_profile": {},
            "retrieved_context": ""
        }

        # 3. Asynchronously execute the node workflow transitions
        final_state = await interview_graph.ainvoke(initial_state)

        # 4. Extract the latest generated message response from the graph engine
        latest_response = final_state["messages"][-1].content

        return {
            "response": latest_response,
            "current_phase": final_state.get("current_phase"),
            "evaluation": final_state.get("evaluation_scores")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph Execution Error: {str(e)}")
