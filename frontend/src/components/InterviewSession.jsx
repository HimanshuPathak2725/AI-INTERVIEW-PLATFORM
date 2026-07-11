import React, { useState, useEffect, useRef } from 'react';
import { sendInterviewResponse } from '../services/interviewApi';

export default function InterviewSession() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: "Great. Let’s dive in. We’ll start with a foundational concept to set the stage." }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [phase, setPhase] = useState('warmup');
  const [score, setScore] = useState(0.1);

  // Ref for automatic scrolling to the latest message
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

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
      setLoading(false);
    }
  };

  const renderMessageContent = (content) => {
    if (!content) return "";
    if (typeof content === 'string') return content;
    if (Array.isArray(content)) {
      return content.map((c) => (typeof c === 'object' ? c.text || JSON.stringify(c) : c)).join("\n");
    }
    if (typeof content === 'object') {
      return content.text || content.content || JSON.stringify(content);
    }
    return String(content);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '80vh', border: '1px solid rgba(148, 163, 184, 0.15)', borderRadius: '20px', overflow: 'hidden', backgroundColor: 'rgba(15, 23, 42, 0.6)', backdropFilter: 'blur(16px)', boxShadow: '0 20px 50px rgba(0,0,0,0.3)' }}>
      
      {/* Header Panel */}
      <div style={{ padding: '20px 24px', background: 'rgba(30, 41, 59, 0.4)', borderBottom: '1px solid rgba(148, 163, 184, 0.12)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ width: '8px', height: '8px', backgroundColor: '#34d399', borderRadius: '50%', display: 'inline-block', boxShadow: '0 0 10px #34d399' }}></span>
            <h4 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>Active Session Console</h4>
          </div>
          <p style={{ margin: '4px 0 0 0', fontSize: '0.78rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
            Current Phase: <strong style={{ color: '#38bdf8' }}>{phase}</strong>
          </p>
        </div>
        
        <div style={{ background: 'linear-gradient(135deg, rgba(15, 118, 110, 0.25), rgba(34, 197, 94, 0.12))', border: '1px solid rgba(45, 212, 191, 0.2)', color: '#a7f3d0', padding: '8px 16px', borderRadius: '14px', fontWeight: 600, fontSize: '0.85rem' }}>
          Accuracy Score: <span style={{ color: '#fff', fontWeight: 800 }}>{(score * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* Messages Viewport */}
      <div style={{ flex: 1, padding: '24px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '18px', background: 'rgba(10, 15, 30, 0.3)' }}>
        {messages.map((msg, idx) => (
          <div key={idx} style={{ display: 'flex', justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
            <div style={{
              maxWidth: '75%',
              padding: '14px 18px',
              borderRadius: msg.role === 'user' ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
              background: msg.role === 'user' ? 'linear-gradient(135deg, #2563eb, #1d4ed8)' : 'rgba(30, 41, 59, 0.7)',
              border: msg.role === 'user' ? '1px solid rgba(59, 130, 246, 0.3)' : '1px solid rgba(148, 163, 184, 0.15)',
              color: msg.role === 'user' ? '#ffffff' : '#e2e8f0',
              fontSize: '0.98rem',
              lineHeight: '1.6',
              whiteSpace: 'pre-line',
              boxShadow: '0 4px 15px rgba(0,0,0,0.1)'
            }}>
              <div style={{ fontSize: '0.7rem', fontWeight: 700, opacity: 0.6, marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                {msg.role === 'user' ? 'You' : 'AI Interviewer'}
              </div>
              {renderMessageContent(msg.content)}
            </div>
          </div>
        ))}
        
        {loading && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#94a3b8', fontStyle: 'italic', fontSize: '0.88rem', paddingLeft: '4px' }}>
            <span style={{ display: 'inline-block', width: '6px', height: '6px', backgroundColor: '#38bdf8', borderRadius: '50%', opacity: 0.7 }}></span>
            Interviewer is evaluating your response...
          </div>
        )}
        
        {/* Anchor point for scrolling */}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Action Bar */}
      <form onSubmit={handleSend} style={{ display: 'flex', padding: '20px 24px', borderTop: '1px solid rgba(148, 163, 184, 0.12)', background: 'rgba(15, 23, 42, 0.8)', gap: '12px', alignItems: 'center' }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Formulate your technical answer..."
          disabled={loading}
          style={{ flex: 1, padding: '14px 20px', borderRadius: '12px', border: '1px solid rgba(148, 163, 184, 0.2)', background: 'rgba(30, 41, 59, 0.5)', color: '#f8fafc', outline: 'none', fontSize: '0.95rem' }}
        />
        <button 
          type="submit" 
          disabled={loading || !input.trim()} 
          style={{ padding: '14px 28px', backgroundColor: '#2563eb', color: '#fff', border: 'none', borderRadius: '12px', cursor: input.trim() && !loading ? 'pointer' : 'not-allowed', fontWeight: 700, fontSize: '0.95rem', opacity: input.trim() && !loading ? 1 : 0.5, transition: 'all 0.2s', boxShadow: '0 4px 12px rgba(37, 99, 235, 0.3)' }}
        >
          Submit
        </button>
      </form>
    </div>
  );
}