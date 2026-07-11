from langgraph.graph import StateGraph, START, END
from backend.app.agents.state import InterviewState
from backend.app.agents.nodes import (
    load_candidate_node,
    retrieval_node,
    interviewer_node,
    evaluation_node,
    finalize_interview_node
)

def route_entry_path(state: InterviewState) -> str:
    """
    Determines if this is the start of the session or an active evaluation turn.
    """
    # If there are 2 or fewer messages, it means it's the opening turn
    if len(state.get("messages", [])) <= 1:
        return "retrieve_context"
    return "evaluate_answer"

def should_continue(state: InterviewState) -> str:
    if state.get("question_count", 0) >= 15 or state.get("current_phase") == "wrap_up":
        return "finalize_interview"
    return "retrieve_context"

workflow = StateGraph(InterviewState)

workflow.add_node("load_candidate", load_candidate_node)
workflow.add_node("retrieve_context", retrieval_node)
workflow.add_node("generate_question", interviewer_node)
workflow.add_node("evaluate_answer", evaluation_node)
workflow.add_node("finalize_interview", finalize_interview_node)

# 1. Start always boots configuration parameters safely
workflow.add_edge(START, "load_candidate")

# 2. Conditional entry routing logic based on current chat progress
workflow.add_conditional_edges(
    "load_candidate",
    route_entry_path,
    {
        "retrieve_context": "retrieve_context",
        "evaluate_answer": "evaluate_answer"
    }
)

# 3. Connection path for evaluation loop
workflow.add_conditional_edges(
    "evaluate_answer",
    should_continue,
    {
        "finalize_interview": "finalize_interview",
        "retrieve_context": "retrieve_context"
    }
)

workflow.add_edge("retrieve_context", "generate_question")
workflow.add_edge("generate_question", END)
workflow.add_edge("finalize_interview", END)

interview_graph = workflow.compile()
