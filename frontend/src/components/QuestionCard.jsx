export default function QuestionCard({ question, answer, onAnswerChange, onSubmit, disabled, index, total }) {
  return (
    <section className="panel question-card">
      <div className="meta">
        <span className="pill">
          Question {index + 1} of {total}
        </span>
        <span className="pill">{question.difficulty}</span>
      </div>

      <h2>{question.question_text}</h2>
      <p className="muted">Expected topics: {(question.expected_topics || []).join(", ") || "general reasoning"}</p>
      {question.context_used ? (
        <details className="context-box">
          <summary>Retrieved context</summary>
          <p>{question.context_used}</p>
        </details>
      ) : null}

      <div className="field">
        <label htmlFor="answer">Your answer</label>
        <textarea
          id="answer"
          value={answer}
          onChange={(event) => onAnswerChange(event.target.value)}
          placeholder="Type your response here..."
        />
      </div>

      <div className="actions">
        <button className="button" type="button" onClick={onSubmit} disabled={disabled}>
          Submit answer
        </button>
      </div>
    </section>
  );
}
