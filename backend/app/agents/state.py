from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class InterviewState(TypedDict):
    # add_messages allows LangGraph to append new conversation chunks automatically
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Metadata tracking the operational state of the interview
    current_phase: str  # e.g., "warmup", "conceptual", "coding", "wrap_up"
    question_count: int
    max_questions_per_phase: int
    
    # Live evaluation metrics updated dynamically after every user response
    evaluation_scores: Dict[str, Any]  # e.g., {"technical_accuracy": 0.0, "communication": 0.0}
    candidate_profile: Dict[str, Any]   # Extracted context from their resume (skills, stack)
    
    # Context injected from our LangChain RAG pipeline based on the topic
    retrieved_context: str
