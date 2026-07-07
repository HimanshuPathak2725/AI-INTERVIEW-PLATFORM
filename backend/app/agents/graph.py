from langgraph.graph import StateGraph, START, END
from backend.app.agents.state import InterviewState
from backend.app.agents.nodes import retrieval_node, evaluation_node, interviewer_node

# 1. Initialize the workflow graph with our custom schema state
workflow = StateGraph(InterviewState)

# 2. Add structural processing nodes to the architecture
workflow.add_node("retrieve_context", retrieval_node)
workflow.add_node("evaluate_response", evaluation_node)
workflow.add_node("generate_question", interviewer_node)

# 3. Establish strict directional operational edges
# When an execution triggers, it follows this straight pipeline line
workflow.add_edge(START, "retrieve_context")
workflow.add_edge("retrieve_context", "evaluate_response")
workflow.add_edge("evaluate_response", "generate_question")
workflow.add_edge("generate_question", END)

# 4. Compile the state machine graph into an executable application binary
interview_graph = workflow.compile()
