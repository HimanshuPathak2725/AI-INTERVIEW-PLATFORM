from sqlalchemy.orm import Session
from typing import Dict, List

from datetime import datetime

from app.models.database import InterviewSession, Question, Answer
from app.models.schemas import InterviewSessionCreate, InterviewSummary
from app.services.rag_pipeline import rag_pipeline

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
    
    def generate_questions_for_session(self, db: Session, session_id: int, num_questions: int = 3) -> List[Question]:
        session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
        if not session:
            raise ValueError("Session not found")
        
        resume_info = session.extracted_skills or {}
        role = session.role
        
        history = self.active_sessions.get(session_id, {}).get("conversation_history", [])
        
        query = rag_pipeline.construct_query(
            resume_info=resume_info,
            role=role,
            conversation_history=history
        )
        
        contexts = rag_pipeline.retrieve_context(query, role, top_k=5)
        generated = rag_pipeline.generate_questions(
            context=contexts,
            resume_info=resume_info,
            role=role,
            num_questions=num_questions,
            conversation_history=history
        )
        
        questions = []
        current_count = db.query(Question).filter(Question.session_id == session_id).count()
        
        for i, q_data in enumerate(generated):
            question = Question(
                session_id=session_id,
                question_text=q_data["question"],
                context_used=q_data.get("context") or "\n\n".join([c["content"] for c in contexts[:2]]),
                expected_topics=q_data.get("expected_concepts", []),
                difficulty=q_data.get("difficulty", "medium"),
                order=current_count + i + 1
            )
            db.add(question)
            questions.append(question)
        
        try:
            db.commit()
            for q in questions:
                db.refresh(q)
        except Exception as e:
            db.rollback()
            raise ValueError(f"Failed to save generated questions: {str(e)}")
        
        if session_id in self.active_sessions:
            for q in generated:
                topic = q.get("topic", "")
                self.active_sessions[session_id]["topics_covered"].append(topic)
                self.active_sessions[session_id]["conversation_history"].append(
                    {
                        "topic": topic,
                        "difficulty": q.get("difficulty", "medium"),
                    }
                )
            self.active_sessions[session_id]["questions_asked"] += len(generated)
        
        return questions
    
    def submit_answer(self, db: Session, session_id: int, question_id: int, answer_text: str) -> Answer:
        question = db.query(Question).filter(Question.id == question_id).first()
        if not question:
            raise ValueError("Question not found")
        
        evaluation = rag_pipeline.evaluate_answer(
            question=question.question_text,
            answer=answer_text,
            expected_concepts=question.expected_topics or []
        )
        
        answer = Answer(
            question_id=question_id,
            answer_text=answer_text,
            score=evaluation.get("score"),
            feedback=evaluation.get("feedback")
        )
        try:
            db.add(answer)
            db.commit()
            db.refresh(answer)
        except Exception as e:
            db.rollback()
            raise ValueError(f"Failed to submit answer: {str(e)}")
        
        return answer
    
    def get_session_questions(self, db: Session, session_id: int) -> List[Question]:
        session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
        if not session:
            raise ValueError("Session not found")
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
        answered = sum(1 for q in questions if q.answers)
        
        scores = []
        strengths = []
        gaps = []
        
        for q in questions:
            if q.answers:
                answer = q.answers[0]
                if answer.score is not None:
                    scores.append(answer.score)
                if answer.feedback:
                    if "good" in answer.feedback.lower() or "excellent" in answer.feedback.lower():
                        strengths.append(q.expected_topics[0] if q.expected_topics else "general")
                    if "missing" in answer.feedback.lower() or "lack" in answer.feedback.lower():
                        gaps.append(q.expected_topics[0] if q.expected_topics else "general")
        
        avg_score = sum(scores) / len(scores) if scores else None
        
        summary = InterviewSummary(
            session_id=session_id,
            role=session.role,
            total_questions=total,
            answered_questions=answered,
            skills_tested=list(set(session.extracted_skills.get("skills", []))),
            average_score=round(avg_score, 2) if avg_score else None,
            strengths=list(set(strengths))[:5],
            gaps=list(set(gaps))[:5],
            overall_feedback=self._generate_overall_feedback(avg_score, strengths, gaps, answered, total),
            generated_report=self._generate_report(session, questions, answered, total, avg_score)
        )
        
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        return summary
    
    def _generate_overall_feedback(self, avg_score, strengths, gaps, answered, total) -> str:
        if avg_score is None:
            return "Session completed. No scoring available."
        
        feedback = f"Overall Score: {avg_score}/100. "
        
        if avg_score >= 80:
            feedback += "Strong performance. Candidate demonstrates solid understanding. "
        elif avg_score >= 60:
            feedback += "Good performance with some areas for improvement. "
        else:
            feedback += "Needs improvement in several areas. "
        
        if strengths:
            feedback += f"Strengths in: {', '.join(strengths[:3])}. "
        if gaps:
            feedback += f"Areas to improve: {', '.join(gaps[:3])}. "
        
        feedback += f"Answered {answered}/{total} questions."
        return feedback

    def _generate_report(self, session, questions, answered, total, avg_score) -> str:
        skills = ", ".join((session.extracted_skills or {}).get("skills", [])[:8]) or "not detected"
        score = "not scored" if avg_score is None else f"{round(avg_score, 2)}/100"
        context_count = sum(1 for question in questions if question.context_used)
        return (
            f"Role: {session.role}. Skills detected: {skills}. "
            f"Answered {answered}/{total} questions. Average score: {score}. "
            f"{context_count} questions include stored retrieval context for traceability."
        )

interview_service = InterviewService()
