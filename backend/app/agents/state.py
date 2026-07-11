from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class InterviewState(TypedDict):
    # Core conversation timeline using LangGraph message addition reducer
    messages: Annotated[List[AnyMessage], add_messages]
    
    # Candidate and Session Profile parameters
    candidate_profile: Dict[str, Any]
    resume_skills: List[str]
    
    # Engine state parameters
    current_phase: str      # warmup, technical, behavioral, wrap_up
    current_topic: str
    covered_topics: List[str]
    difficulty: str         # easy, medium, hard
    question_count: int
    
    # Security and Duplicate Prevention tracking datasets
    asked_questions: List[str]
    question_hashes: List[str]
    
    # Retrieval and Evaluation Telemetry maps
    retrieved_context: str
    evaluation_scores: Dict[str, Any]
    last_feedback: str
    conversation_summary: str
