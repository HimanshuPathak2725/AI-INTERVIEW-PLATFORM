import React, { useState } from "react";

async function createSession(formData) {
  const response = await fetch("/api/v1/sessions", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) throw new Error(`Failed to create session: ${response.status}`);
  return response.json();
}

async function generateQuestions(sessionId) {
  const response = await fetch(`/api/v1/sessions/${sessionId}/questions?num_questions=3`, {
    method: "POST",
  });
  if (!response.ok) throw new Error(`Failed to generate questions: ${response.status}`);
  return response.json();
}

export default function ResumeUpload({ onSessionCreated }) {
  const [candidateName, setCandidateName] = useState("");
  const [role, setRole] = useState("backend engineer");
  const [resumeText, setResumeText] = useState("");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("role", role);
      if (candidateName.trim()) formData.append("candidate_name", candidateName.trim());
      if (resumeText.trim()) formData.append("resume_text", resumeText.trim());
      if (file) formData.append("resume_file", file);

      const session = await createSession(formData);
      const questions = await generateQuestions(session.id);
      onSessionCreated({ ...session, questions });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} style={{ padding: '32px', borderRadius: '24px', background: 'rgba(15, 23, 42, 0.75)', border: '1px solid rgba(148, 163, 184, 0.16)', color: '#e5e7eb', display: 'grid', gap: '24px', boxShadow: '0 25px 60px rgba(0,0,0,0.35)' }}>
      <div>
        <h2 style={{ fontSize: '1.8rem', fontWeight: 800, color: '#f8fafc', margin: '0 0 6px 0' }}>Start screening session</h2>
        <p style={{ margin: 0, fontSize: '0.9rem', color: '#94a3b8' }}>Use a resume file or pasted text to seed the evaluation vector.</p>
      </div>
      
      <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: '240px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <label htmlFor="candidate-name" style={{ fontSize: '0.82rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase' }}>Candidate name</label>
          <input
            id="candidate-name"
            value={candidateName}
            onChange={(event) => setCandidateName(event.target.value)}
            placeholder="Enter candidate name"
            style={{ padding: '12px 16px', borderRadius: '12px', border: '1px solid rgba(148, 163, 184, 0.2)', background: 'rgba(30, 41, 59, 0.5)', color: '#f8fafc', outline: 'none' }}
          />
        </div>

        <div style={{ flex: 1, minWidth: '240px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <label htmlFor="role" style={{ fontSize: '0.82rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase' }}>Target Role Profile</label>
          <select 
            id="role" 
            value={role} 
            onChange={(event) => setRole(event.target.value)}
            style={{ padding: '12px 16px', borderRadius: '12px', border: '1px solid rgba(148, 163, 184, 0.2)', background: '#1e293b', color: '#f8fafc', outline: 'none', cursor: 'pointer' }}
          >
            <option value="backend engineer">Backend Engineer</option>
            <option value="ai ml engineer">AI / ML Engineer</option>
            <option value="frontend engineer">Frontend Engineer</option>
          </select>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <label htmlFor="resume-file" style={{ fontSize: '0.82rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase' }}>Resume Passing Core (PDF/TXT)</label>
        <input
          id="resume-file"
          type="file"
          accept=".pdf,.txt"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          style={{ padding: '10px', borderRadius: '12px', border: '1px dashed rgba(148, 163, 184, 0.3)', background: 'rgba(30, 41, 59, 0.2)', color: '#cbd5e1', cursor: 'pointer' }}
        />
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <label htmlFor="resume-text" style={{ fontSize: '0.82rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase' }}>Optional resume raw text</label>
        <textarea
          id="resume-text"
          value={resumeText}
          onChange={(event) => setResumeText(event.target.value)}
          placeholder="Paste plain resume structural contents if file upload is bypassed..."
          style={{ minHeight: '90px', padding: '12px 16px', borderRadius: '12px', border: '1px solid rgba(148, 163, 184, 0.2)', background: 'rgba(30, 41, 59, 0.5)', color: '#f8fafc', outline: 'none', resize: 'vertical' }}
        />
      </div>

      {error ? (
        <p style={{ margin: 0, fontSize: '0.88rem', color: '#f87171', fontWeight: 500 }}>⚠️ {error}</p>
      ) : (
        <p style={{ margin: 0, fontSize: '0.85rem', color: '#64748b', fontStyle: 'italic' }}>Detected contextual weights and parsing matrices will synchronize automatically.</p>
      )}

      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '8px' }}>
        <button 
          type="submit" 
          disabled={loading}
          style={{ padding: '14px 28px', borderRadius: '12px', border: 'none', background: 'linear-gradient(135deg, #10b981, #059669)', color: '#fff', fontWeight: 700, cursor: loading ? 'not-allowed' : 'pointer', transition: 'all 0.2s', boxShadow: '0 4px 14px rgba(16, 185, 129, 0.3)' }}
        >
          {loading ? "Initializing Runtime Engine..." : "Launch Evaluation System"}
        </button>
      </div>
    </form>
  );
}