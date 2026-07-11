from sqlalchemy.orm import Session
from typing import Dict, List, Any
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

# Explicitly mapping imports to root PYTHONPATH boundary
from backend.app.models.database import InterviewSession, Question, Answer
from backend.app.models.schemas import InterviewSessionCreate, InterviewSummary
from backend.app.agents.graph import interview_graph

class InterviewService:
    def __init__(self):
        self.active_sessions: Dict[int, Dict] = {}
    
    def create_session(self, db: Session, session_data: InterviewSessionCreate) -> InterviewSession:
        session = InterviewSession(
            candidate_name=session_data.candidate_name,
            role=session_data.role,
            resume_text=session_data.resume_text,
            extracted_skills=session_data.resume_info.model_dump() if session_data.resume_info else {},
            status="active"
        )
        try:
            db.add(session)
            db.commit()
            db.refresh(session)
        except Exception as e:
            db.rollback()
            raise ValueError(f"Failed to create session: {str(e)}")
            
        self.active_sessions[session.id] = {
            "questions_asked": 0,
            "topics_covered": [],
            "conversation_history": []
        }
        return session

    async def process_interview_turn(self, db: Session, session_id: int, user_message: str) -> str:
        """
        Orchestrates a complete stateful turn using LangGraph. Replaces the legacy 
        procedural generation steps with unified state chart execution.
        """
        session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
        if not session:
            raise ValueError("Interview session not found")

        # 1. Reconstruct chat timeline from Database to feed LangGraph state
        db_questions = db.query(Question).filter(Question.session_id == session_id).order_by(Question.order).all()
        
        messages_timeline = []
        asked_questions_text = []
        
        for q in db_questions:
            messages_timeline.append(AIMessage(content=q.question_text))
            asked_questions_text.append(q.question_text)
            # Find matching answer if candidate responded
            ans = db.query(Answer).filter(Answer.question_id == q.id).first()
            if ans:
                messages_timeline.append(HumanMessage(content=ans.answer_text))

        # Append the current incoming user response
        messages_timeline.append(HumanMessage(content=user_message))

        # 2. Extract current technical scores evaluation fallback
        current_scores = {"technical_accuracy": 0.1}
        if db_questions:
            last_q = db_questions[-1]
            last_ans = db.query(Answer).filter(Answer.question_id == last_q.id).first()
            if last_ans and last_ans.score is not None:
                current_scores["technical_accuracy"] = float(last_ans.score) / 100.0

        # 3. Build initial LangGraph graph frame state payload
        graph_input = {
            "messages": messages_timeline,
            "candidate_profile": session.extracted_skills or {},
            "resume_skills": (session.extracted_skills or {}).get("skills", []),
            "current_phase": "technical" if len(db_questions) > 1 else "warmup",
            "current_topic": "System Design & Performance" if len(db_questions) > 1 else "General Introduction",
            "covered_topics": [q.expected_topics[0] for q in db_questions if q.expected_topics] or [],
            "difficulty": "medium" if len(db_questions) > 2 else "easy",
            "question_count": len(db_questions),
            "asked_questions": asked_questions_text,
            "question_hashes": [],
            "retrieved_context": "",
            "evaluation_scores": current_scores,
            "last_feedback": "",
            "conversation_summary": ""
        }

        # 4. Trigger production routing graph agent execution loop
        final_state = await interview_graph.ainvoke(graph_input)
        
        # 5. Extract newly generated response from final output timeline
        ai_response = final_state["messages"][-1].content
        
        # 6. Save the previous answer score if processed by evaluation node
        if db_questions and "evaluation_scores" in final_state:
            latest_q = db_questions[-1]
            existing_ans = db.query(Answer).filter(Answer.question_id == latest_q.id).first()
            if not existing_ans:
                # Save user's current answer text to database linked to the last question
                new_ans = Answer(
                    question_id=latest_q.id,
                    answer_text=user_message,
                    score=int(final_state["evaluation_scores"].get("technical_accuracy", 0.1) * 100),
                    feedback=final_state.get("last_feedback", "Evaluated via LangGraph Node Architecture.")
                )
                db.add(new_ans)

        # 7. Detect final scorecard state and skip persisting it as a Question
        # Final reports are identified by conversation_summary or wrap_up phase
        is_final_scorecard = (
            final_state.get("conversation_summary")
            or final_state.get("current_phase") == "wrap_up"
        )

        if not is_final_scorecard:
            # Persist the newly generated question into DB to lock state
            new_db_question = Question(
                session_id=session_id,
                question_text=ai_response,
                context_used=final_state.get("retrieved_context", "Direct Context Matrix"),
                expected_topics=[final_state.get("current_topic", "Technical Stack")],
                difficulty=final_state.get("difficulty", "easy"),
                order=len(db_questions) + 1
            )

            try:
                db.add(new_db_question)
                db.commit()
            except Exception as e:
                db.rollback()
                raise ValueError(f"Failed to synchronize state tracking with database: {str(e)}")
        else:
            # Persist final report through session summary (commit answer changes only)
            try:
                db.commit()
            except Exception as e:
                db.rollback()
                raise ValueError(f"Failed to save final evaluation data: {str(e)}")

        return ai_response

    def get_session_questions(self, db: Session, session_id: int) -> List[Question]:
        return db.query(Question).filter(Question.session_id == session_id).order_by(Question.order).all()

    def complete_session(self, db: Session, session_id: int) -> InterviewSummary:
        session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
        if not session:
            raise ValueError("Session not found")

        session.status = "completed"
        session.completed_at = datetime.utcnow()
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            raise ValueError(f"Failed to complete session: {str(e)}")

        questions = self.get_session_questions(db, session_id)
        total = len(questions)

        # Aggregate actual evaluation data from Answer results
        answers = []
        scores = []
        all_feedback = []

        for q in questions:
            ans = db.query(Answer).filter(Answer.question_id == q.id).first()
            if ans:
                answers.append(ans)
                if ans.score is not None:
                    scores.append(ans.score)
                if ans.feedback:
                    all_feedback.append(ans.feedback)

        answered = len(answers)

        # Derive average_score from actual stored scores
        average_score = sum(scores) / len(scores) if scores else 0.0

        # Extract strengths and gaps from feedback patterns
        # Simple heuristic: look for positive and negative indicators
        strengths = []
        gaps = []
        for feedback in all_feedback:
            feedback_lower = feedback.lower()
            if any(word in feedback_lower for word in ["strong", "excellent", "good", "solid", "clear"]):
                strengths.append(feedback[:100])  # Truncate for brevity
            if any(word in feedback_lower for word in ["missing", "weak", "gap", "improve", "lack", "insufficient"]):
                gaps.append(feedback[:100])

        # Remove duplicates and limit to reasonable count
        strengths = list(dict.fromkeys(strengths))[:5]
        gaps = list(dict.fromkeys(gaps))[:5]

        # If no specific strengths/gaps found, provide generic feedback
        if not strengths:
            strengths = ["Completed interview session"] if answered > 0 else []
        if not gaps:
            gaps = ["Further evaluation needed"] if answered < total else []

        # Derive overall feedback from average performance
        if average_score >= 80:
            overall_feedback = "Strong performance across technical assessments."
        elif average_score >= 60:
            overall_feedback = "Solid understanding with room for improvement in specific areas."
        elif average_score >= 40:
            overall_feedback = "Partial understanding demonstrated; additional preparation recommended."
        else:
            overall_feedback = "Significant gaps identified in technical knowledge."

        # Generate report summary
        generated_report = (
            f"# Interview Summary for {session.role}\n\n"
            f"**Questions Asked:** {total}\n"
            f"**Questions Answered:** {answered}\n"
            f"**Average Score:** {average_score:.1f}%\n\n"
            f"## Strengths\n" + "\n".join([f"- {s}" for s in strengths]) + "\n\n"
            f"## Areas for Improvement\n" + "\n".join([f"- {g}" for g in gaps]) + "\n\n"
            f"## Overall Assessment\n{overall_feedback}"
        )

        summary = InterviewSummary(
            session_id=session_id,
            role=session.role,
            total_questions=total,
            answered_questions=answered,
            skills_tested=list(set((session.extracted_skills or {}).get("skills", []))),
            average_score=average_score,
            strengths=strengths,
            gaps=gaps,
            overall_feedback=overall_feedback,
            generated_report=generated_report
        )
        return summary

interview_service = InterviewService()
