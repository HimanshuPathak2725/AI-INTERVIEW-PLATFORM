import { useMemo, useState } from "react";

import QuestionCard from "./QuestionCard";

async function submitAnswer(sessionId, questionId, answerText) {
  const response = await fetch("/api/v1/answers", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      question_id: questionId,
      answer_text: answerText,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to submit answer: ${response.status}`);
  }

  return response.json();
}

async function completeSession(sessionId) {
  const response = await fetch(`/api/v1/sessions/${sessionId}/complete`, {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(`Failed to complete session: ${response.status}`);
  }

  return response.json();
}

export default function InterviewSession({ session, onComplete }) {
  const questions = useMemo(() => session.questions ?? [], [session.questions]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answer, setAnswer] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const currentQuestion = questions[currentIndex];

  const handleSubmit = async () => {
    if (!currentQuestion || !answer.trim()) {
      setError("Please write an answer before submitting.");
      return;
    }

    setBusy(true);
    setError("");

    try {
      await submitAnswer(session.id, currentQuestion.id, answer.trim());
      setAnswer("");

      if (currentIndex + 1 < questions.length) {
        setCurrentIndex((value) => value + 1);
        return;
      }

      const summary = await completeSession(session.id);
      onComplete(summary);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  };

  if (!currentQuestion) {
    return (
      <section className="panel">
        <h2>No questions available yet.</h2>
        <p className="muted">The backend returned an empty question set for this session.</p>
      </section>
    );
  }

  return (
    <div className="grid">
      <section className="panel">
        <div className="meta">
          <span className="pill">{session.role}</span>
          <span className="pill">{session.candidate_name || "Anonymous candidate"}</span>
        </div>
        <p className="muted">Answer the generated interview questions one by one.</p>
      </section>

      <QuestionCard
        question={currentQuestion}
        answer={answer}
        onAnswerChange={setAnswer}
        onSubmit={handleSubmit}
        disabled={busy}
        index={currentIndex}
        total={questions.length}
      />

      {error ? <p className="panel muted">{error}</p> : null}
    </div>
  );
}
