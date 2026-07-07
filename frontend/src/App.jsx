import React, { useState, useEffect, useRef } from 'react';
import { fetchInterviewResponse, uploadResumeFile } from './services/interviewApi';

function App() {
  const [isUploaded, setIsUploaded] = useState(false);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Welcome to your AI Technical Interview. I have analyzed your profile context. Let’s begin.' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [phase, setPhase] = useState('warmup');
  const [score, setScore] = useState(0.0);
  
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUploadSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setUploading(true);
    try {
      await uploadResumeFile(file);
      setIsUploaded(true);
    } catch (err) {
      alert("Failed to parse resume. Ensure backend is up and running.");
    } finally {
      setUploading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userText = input;
    setInput('');
    
    const updatedMessages = [...messages, { role: 'user', content: userText }];
    setMessages(updatedMessages);
    setLoading(true);

    try {
      const apiHistory = updatedMessages.map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      const result = await fetchInterviewResponse(userText, apiHistory, phase);
      setPhase(result.nextPhase);
      setScore(result.evaluation.technical_accuracy);
      setMessages(prev => [...prev, { role: 'assistant', content: result.aiMessage }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Failed to sync with evaluation engine.' }]);
    } finally {
      setLoading(false);
    }
  };

  // 1. Initial State: Show Resume Upload View
  if (!isUploaded) {
    return (
      <div style={{ backgroundColor: '#0f172a', minHeight: '100vh', color: '#f8fafc', fontFamily: 'sans-serif', display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '20px' }}>
        <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '12px', padding: '40px', maxWidth: '480px', width: '100%', textAlign: 'center', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}>
          <h2 style={{ color: '#38bdf8', marginBottom: '8px' }}>Upload Your Resume</h2>
          <p style={{ color: '#94a3b8', fontSize: '0.9rem', marginBottom: '24px' }}>Please upload your technical resume in PDF format to configure the personalized AI interview environment.</p>
          
          <form onSubmit={handleUploadSubmit}>
            <input 
              type="file" 
              accept=".pdf" 
              onChange={handleFileChange} 
              style={{ display: 'none' }} 
              id="resume-file-input"
            />
            <label htmlFor="resume-file-input" style={{ display: 'block', padding: '24px', border: '2px dashed #334155', borderRadius: '8px', cursor: 'pointer', marginBottom: '20px', backgroundColor: '#0f172a', color: '#94a3b8', transition: 'border-color 0.2s' }} onMouseOver={(e) => e.target.style.borderColor = '#0284c7'} onMouseOut={(e) => e.target.style.borderColor = '#334155'}>
              {file ? file.name : "Click to browse or drop PDF here"}
            </label>
            <button 
              type="submit" 
              disabled={uploading || !file} 
              style={{ width: '100%', padding: '14px', backgroundColor: uploading || !file ? '#334155' : '#0284c7', color: '#fff', border: 'none', borderRadius: '8px', fontWeight: 'bold', cursor: uploading || !file ? 'not-allowed' : 'pointer' }}
            >
              {uploading ? "Analyzing Resume Context..." : "Initialize Session"}
            </button>
          </form>
        </div>
      </div>
    );
  }

  // 2. Active State: Show Premium Interview Console
  return (
    <div style={{ backgroundColor: '#0f172a', minHeight: '100vh', color: '#f8fafc', fontFamily: 'system-ui, sans-serif', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '24px' }}>
      <header style={{ width: '100%', maxWidth: '900px', background: '#1e293b', borderRadius: '12px', padding: '16px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', border: '1px solid #334155', marginBottom: '20px' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '1.25rem', color: '#38bdf8' }}>AI Interview Console 🚀</h2>
          <p style={{ margin: '4px 0 0 0', fontSize: '0.85rem', color: '#94a3b8' }}>Current Phase: <span style={{ color: '#f43f5e', fontWeight: 'bold' }}>{phase}</span></p>
        </div>
        <div style={{ background: 'rgba(34, 197, 94, 0.15)', border: '1px solid #22c55e', color: '#4ade80', padding: '8px 16px', borderRadius: '8px', fontWeight: '700', fontSize: '0.95rem' }}>
          <div style={{ fontSize: '0.75rem', color: '#94a3b8', fontWeight: 'normal' }}>Technical Accuracy</div>
          {(score * 100).toFixed(0)}%
        </div>
      </header>

      <main style={{ width: '100%', maxWidth: '900px', flex: 1, height: '55vh', background: '#1e293b', border: '1px solid #334155', borderRadius: '12px', padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px', overflowY: 'auto' }}>
        {messages.map((msg, i) => (
          <div key={i} style={{ display: 'flex', justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
            <div style={{ maxWidth: '75%', padding: '12px 18px', borderRadius: '12px', backgroundColor: msg.role === 'user' ? '#0284c7' : '#334155', color: '#f8fafc', borderBottomRightRadius: msg.role === 'user' ? '2px' : '12px', borderBottomLeftRadius: msg.role === 'user' ? '12px' : '2px' }}>
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
            style={{ flex: 1, padding: '14px 18px', borderRadius: '10px', border: '1px solid #334155', backgroundColor: '#1e293b', color: '#f8fafc', outline: 'none' }}
          />
          <button type="submit" disabled={loading || !input.trim()} style={{ padding: '14px 28px', backgroundColor: loading || !input.trim() ? '#334155' : '#0284c7', color: '#fff', border: 'none', borderRadius: '10px', fontWeight: '600', cursor: 'pointer' }}>
            Send
          </button>
        </form>
      </footer>
    </div>
  );
}

export default App;