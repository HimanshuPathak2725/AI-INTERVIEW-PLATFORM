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
    # If there are 2 or fewer messages, it means it's the opening turn.
    if len(state.get("messages", [])) <= 1:
        return "retrieve_context"
    return "evaluate_answer"


def assessment_router(state: InterviewState) -> str:
    """
    Deterministically updates the interview phase using the total number of assistant questions.
    The router runs after question generation so the next turn starts in the correct phase.
    Finalizes only when 5 questions have been answered, not when the 5th question is generated.
    """
    messages = state.get("messages", [])
    total_ai_questions = sum(
        1
        for message in messages
        if getattr(message, "type", "") == "ai"
        or getattr(message, "role", "") == "assistant"
    )

    # Count human responses to determine how many questions have been answered
    total_human_responses = sum(
        1
        for message in messages
        if getattr(message, "type", "") == "human"
        or getattr(message, "role", "") == "user"
    )

    # Finalize only when 5 prior questions have been answered (5 AI questions + 5 human answers)
    # This ensures the 5th question remains available for the candidate to answer
    if total_ai_questions >= 5 and total_human_responses >= 5:
        state["current_phase"] = "wrap_up"
        return "finalize_interview"

    if total_ai_questions < 2:
        state["current_phase"] = "warmup"
    elif total_ai_questions in (2, 3):
        state["current_phase"] = "deep_dive"
    elif total_ai_questions == 4:
        state["current_phase"] = "system_design"

    return "end"

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

# 3. Evaluation always precedes the next retrieval/generation cycle.
workflow.add_edge("evaluate_answer", "retrieve_context")

workflow.add_edge("retrieve_context", "generate_question")
workflow.add_conditional_edges(
    "generate_question",
    assessment_router,
    {
        "finalize_interview": "finalize_interview",
        "end": END,
    }
)
workflow.add_edge("finalize_interview", END)

interview_graph = workflow.compile()
