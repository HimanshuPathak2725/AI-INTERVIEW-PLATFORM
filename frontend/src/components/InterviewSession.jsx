import React, { useState } from 'react';
import { sendInterviewResponse } from '../services/interviewApi'; // Updated to use correct stable service

export default function InterviewSession() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: "Great. Let’s dive in. We’ll start with a foundational concept to set the stage." }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [phase, setPhase] = useState('warmup');
  const [score, setScore] = useState(0.1);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const currentInput = input;
    setInput('');

    const freshMessages = [...messages, { role: 'user', content: currentInput }];
    setMessages(freshMessages);
    setLoading(true);

    try {
      const result = await sendInterviewResponse(currentInput, freshMessages, phase);
      
      setPhase(result.nextPhase || result.current_phase || phase);
      if (result.evaluation?.technical_accuracy !== undefined) {
        setScore(result.evaluation.technical_accuracy);
      }
      
      setMessages(prev => [...prev, { role: 'assistant', content: result.aiMessage || result.response }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Connection timed out or backend offline.' }]);
    } finally {
      loading && setLoading(false);
    }
  };

  // Helper helper to render content safely without object injection crashes
  const renderMessageContent = (content) => {
    if (!content) return "";
    if (typeof content === 'string') return content;
    if (Array.isArray(content)) {
      return content.map((c, i) => (typeof c === 'object' ? c.text || JSON.stringify(c) : c)).join("\n");
    }
    if (typeof content === 'object') {
      return content.text || content.content || JSON.stringify(content);
    }
    return String(content);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '75vh', border: '1px solid #e0e0e0', borderRadius: '8px', overflow: 'hidden', backgroundColor: '#fff' }}>
      <div style={{ padding: '16px', background: '#f8f9fa', borderBottom: '1px solid #e0e0e0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h4 style={{ margin: 0 }}>Active Session Console</h4>
          <small style={{ color: '#6c757d' }}>Phase: <strong>{phase}</strong></small>
        </div>
        <div style={{ background: '#e2f0d9', color: '#385723', padding: '6px 12px', borderRadius: '20px', fontWeight: 'bold', fontSize: '0.9rem' }}>
          Accuracy Score: {(score * 100).toFixed(0)}%
        </div>
      </div>

      <div style={{ flex: 1, padding: '20px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {messages.map((msg, idx) => (
          <div key={idx} style={{ display: 'flex', justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
            <div style={{
              maxWidth: '70%',
              padding: '12px 16px',
              borderRadius: '12px',
              backgroundColor: msg.role === 'user' ? '#007bff' : '#f1f3f5',
              color: msg.role === 'user' ? '#fff' : '#212529',
              whiteSpace: 'pre-line'
            }}>
              {renderMessageContent(msg.content)}
            </div>
          </div>
        ))}
        {loading && <div style={{ color: '#6c757d', fontStyle: 'italic', fontSize: '0.85rem' }}>Interviewer is evaluating your response...</div>}
      </div>

      <form onSubmit={handleSend} style={{ display: 'flex', padding: '16px', borderTop: '1px solid #e0e0e0', background: '#f8f9fa', gap: '10px' }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Formulate your technical answer..."
          disabled={loading}
          style={{ flex: 1, padding: '12px 16px', borderRadius: '6px', border: '1px solid #ced4da', outline: 'none' }}
        />
        <button type="submit" disabled={loading} style={{ padding: '12px 24px', backgroundColor: '#007bff', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold' }}>
          Submit
        </button>
      </form>
    </div>
  );
}
