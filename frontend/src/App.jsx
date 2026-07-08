import React, { useState, useRef, useEffect } from 'react';
import { sendInterviewResponse, uploadResume } from './services/interviewApi';

export default function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: "Great. Let’s dive in. We’ll start with a foundational concept to set the stage." }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [phase, setPhase] = useState('warmup');
  const [score, setScore] = useState(0.1);
  const [uploadStatus, setUploadStatus] = useState('');
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleResumeUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setUploadStatus('Uploading resume...');
    try {
      const data = await uploadResume(file);
      setUploadStatus('✅ Resume parsed and context loaded!');
      if (data.message) {
        setMessages(prev => [...prev, { role: 'assistant', content: data.message }]);
      }
    } catch (err) {
      console.error(err);
      setUploadStatus('❌ Upload failed. Please check file type.');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const currentInput = input;
    setInput('');

    // Maintain strict flat string entries inside state array
    const freshMessages = [...messages, { role: 'user', content: currentInput }];
    setMessages(freshMessages);
    setLoading(true);

    try {
      const result = await sendInterviewResponse(currentInput, freshMessages, phase);
      
      setPhase(result.current_phase || phase);
      if (result.evaluation?.technical_accuracy !== undefined) {
        setScore(result.evaluation.technical_accuracy);
      }
      
      // Backend guarantees a clean text string now
      const aiResponse = result.response || "No structured response received.";
      setMessages(prev => [...prev, { role: 'assistant', content: aiResponse }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Connection timed out or state schema mismatch.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', backgroundColor: '#0f172a', color: '#f8fafc', fontFamily: 'sans-serif', padding: '20px' }}>
      <header style={{ width: '100%', maxWidth: '900px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', borderBottom: '1px solid #334155', paddingBottom: '16px' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '1.25rem', color: '#38bdf8' }}>AI Interview Console 🚀</h2>
          <p style={{ margin: '4px 0 0 0', fontSize: '0.85rem', color: '#94a3b8' }}>Current Phase: <span style={{ color: '#f43f5e', fontWeight: 'bold' }}>{phase}</span></p>
        </div>
        <div style={{ background: 'rgba(34, 197, 94, 0.15)', border: '1px solid #22c55e', color: '#4ade80', padding: '8px 16px', borderRadius: '8px', fontWeight: '700', fontSize: '0.95rem' }}>
          <div style={{ fontSize: '0.75rem', color: '#94a3b8', fontWeight: 'normal' }}>Technical Accuracy</div>
          {(score * 100).toFixed(0)}%
        </div>
      </header>

      {/* Resume Upload Module */}
      <div style={{ width: '100%', maxWidth: '900px', background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', padding: '16px', marginBottom: '16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px' }}>
        <div>
          <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 'bold', color: '#94a3b8', marginBottom: '6px' }}>Upload Candidate Resume (PDF/TXT)</label>
          <input type="file" accept=".pdf,.txt" onChange={handleResumeUpload} style={{ fontSize: '0.85rem', color: '#f8fafc' }} />
        </div>
        <span style={{ fontSize: '0.85rem', color: '#38bdf8', fontStyle: 'italic' }}>{uploadStatus}</span>
      </div>

      <main style={{ width: '100%', maxWidth: '900px', flex: 1, height: '50vh', background: '#1e293b', border: '1px solid #334155', borderRadius: '12px', padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px', overflowY: 'auto' }}>
        {messages.map((msg, i) => (
          <div key={i} style={{ display: 'flex', justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
            <div style={{ maxWidth: '75%', padding: '12px 18px', borderRadius: '12px', backgroundColor: msg.role === 'user' ? '#0284c7' : '#334155', color: '#f8fafc', borderBottomRightRadius: msg.role === 'user' ? '2px' : '12px', borderBottomLeftRadius: msg.role === 'user' ? '12px' : '2px', lineHeight: '1.5', whiteSpace: 'pre-wrap' }}>
              {msg.content}
            </div>
          </div>
        ))}
        {loading && <div style={{ color: '#94a3b8', fontSize: '0.85rem', fontStyle: 'italic' }}>Interviewer is assessing your response...</div>}
        <div ref={chatEndRef} />
      </main>

      <footer style={{ width: '100%', maxWidth: '900px', marginTop: '16px' }}>
        <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '12px' }}>
          <input 
            type="text" 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your structured technical answer..."
            disabled={loading}
            style={{ flex: 1, padding: '14px 20px', borderRadius: '8px', border: '1px solid #475569', backgroundColor: '#1e293b', color: '#f8fafc', outline: 'none', fontSize: '0.95rem' }}
          />
          <button type="submit" disabled={loading} style={{ padding: '14px 28px', backgroundColor: '#0284c7', color: '#fff', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold', fontSize: '0.95rem' }}>
            Send
          </button>
        </form>
      </footer>
    </div>
  );
}
