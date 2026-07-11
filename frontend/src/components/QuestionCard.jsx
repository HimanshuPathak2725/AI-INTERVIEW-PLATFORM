import React from 'react';

export default function QuestionCard({ question, answer, onAnswerChange, onSubmit, disabled, index, total }) {
  return (
    <section style={{ padding: '28px', borderRadius: '20px', background: 'rgba(15, 23, 42, 0.75)', border: '1px solid rgba(148, 163, 184, 0.16)', color: '#e5e7eb', boxShadow: '0 20px 50px rgba(0,0,0,0.2)' }}>
      <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
        <span style={{ padding: '4px 10px', borderRadius: '8px', fontSize: '0.75rem', fontWeight: 600, background: 'rgba(56, 189, 248, 0.15)', color: '#38bdf8', border: '1px solid rgba(56, 189, 248, 0.2)' }}>
          Question {index + 1} of {total}
        </span>
        <span style={{ padding: '4px 10px', borderRadius: '8px', fontSize: '0.75rem', fontWeight: 600, background: 'rgba(245, 158, 11, 0.15)', color: '#fbbf24', border: '1px solid rgba(245, 158, 11, 0.2)', textTransform: 'uppercase' }}>
          {question.difficulty || 'Intermediate'}
        </span>
      </div>

      <h2 style={{ fontSize: '1.6rem', fontWeight: 700, color: '#f8fafc', margin: '0 0 10px 0', lineHeight: '1.3' }}>{question.question_text}</h2>
      <p style={{ margin: '0 0 20px 0', fontSize: '0.88rem', color: '#94a3b8' }}>
        Expected topics: <span style={{ color: '#cbd5e1', fontWeight: 500 }}>{(question.expected_topics || []).join(", ") || "general reasoning"}</span>
      </p>

      {question.context_used ? (
        <details style={{ marginBottom: '24px', background: 'rgba(30, 41, 59, 0.4)', border: '1px solid rgba(148, 163, 184, 0.1)', borderRadius: '12px', padding: '12px' }}>
          <summary style={{ cursor: 'pointer', fontSize: '0.88rem', color: '#38bdf8', fontWeight: 600, outline: 'none' }}>Retrieved context matrix</summary>
          <p style={{ margin: '10px 0 0 0', fontSize: '0.9rem', color: '#cbd5e1', lineHeight: '1.6' }}>{question.context_used}</p>
        </details>
      ) : null}

      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '20px' }}>
        <label htmlFor="answer" style={{ fontSize: '0.88rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Your answer</label>
        <textarea
          id="answer"
          value={answer}
          onChange={(event) => onAnswerChange(event.target.value)}
          placeholder="Type your structured response here..."
          style={{ minHeight: '120px', padding: '14px', borderRadius: '12px', border: '1px solid rgba(148, 163, 184, 0.2)', background: 'rgba(30, 41, 59, 0.5)', color: '#f8fafc', outline: 'none', resize: 'vertical', fontSize: '0.95rem', lineHeight: '1.6' }}
        />
      </div>

      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <button 
          type="button" 
          onClick={onSubmit} 
          disabled={disabled}
          style={{ padding: '12px 24px', borderRadius: '12px', border: 'none', background: '#2563eb', color: '#fff', fontWeight: 700, cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.5 : 1, boxShadow: '0 4px 12px rgba(37, 99, 235, 0.25)' }}
        >
          Submit answer
        </button>
      </div>
    </section>
  );
}