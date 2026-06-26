from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Optional, List
from app.database import get_db
from app.models.schemas import (
    InterviewSessionCreate, InterviewSessionResponse, QuestionResponse,
    AnswerSubmit, AnswerResponse, InterviewSummary, ResumeInfo
)
from app.services.resume_parser import resume_parser_service
from app.services.interview_service import interview_service
from app.models.database import InterviewSession, Question, Answer

router = APIRouter(prefix="/api/v1", tags=["interview"])

@router.post("/sessions", response_model=InterviewSessionResponse)
async def create_session(
    role: str = Form(...),
    candidate_name: Optional[str] = Form(None),
    resume_text: Optional[str] = Form(None),
    resume_file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    resume_info = None

    if resume_file:
        contents = await resume_file.read()
        resume_text, resume_info = resume_parser_service.parse(
            contents, resume_file.filename
        )
    elif resume_text:
        resume_text, resume_info = resume_parser_service.parse_text(resume_text)

    session_data = InterviewSessionCreate(
        candidate_name=candidate_name,
        role=role,
        resume_text=resume_text,
        resume_info=resume_info
    )
    
    session = interview_service.create_session(db, session_data)
    return session

@router.post("/sessions/{session_id}/questions", response_model=List[QuestionResponse])
def generate_questions(
    session_id: int,
    num_questions: int = 3,
    db: Session = Depends(get_db)
):
    try:
        return interview_service.generate_questions_for_session(db, session_id, num_questions)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@router.get("/sessions/{session_id}/questions", response_model=List[QuestionResponse])
def get_session_questions(session_id: int, db: Session = Depends(get_db)):
    try:
        return interview_service.get_session_questions(db, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@router.post("/answers", response_model=AnswerResponse)
def submit_answer(answer_data: AnswerSubmit, db: Session = Depends(get_db)):
    try:
        return interview_service.submit_answer(
            db, answer_data.session_id, answer_data.question_id, answer_data.answer_text
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@router.post("/sessions/{session_id}/complete", response_model=InterviewSummary)
def complete_session(session_id: int, db: Session = Depends(get_db)):
    try:
        return interview_service.complete_session(db, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@router.get("/sessions/{session_id}/summary")
def get_session_summary(session_id: int, db: Session = Depends(get_db)):
    session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    questions = db.query(Question).filter(Question.session_id == session_id).all()
    answers = db.query(Answer).join(Question).filter(Question.session_id == session_id).all()
    
    return {
        "session_id": session_id,
        "role": session.role,
        "status": session.status,
        "total_questions": len(questions),
        "answered_questions": len(answers),
        "questions": [
            {
                "id": q.id,
                "question": q.question_text,
                "difficulty": q.difficulty,
                "has_answer": bool(q.answers)
            }
            for q in questions
        ]
    }

@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "ai-interview-platform"}
