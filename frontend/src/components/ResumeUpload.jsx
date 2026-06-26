import { useState } from "react";

async function createSession(formData) {
  const response = await fetch("/api/v1/sessions", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Failed to create session: ${response.status}`);
  }

  return response.json();
}

async function generateQuestions(sessionId) {
  const response = await fetch(`/api/v1/sessions/${sessionId}/questions?num_questions=3`, {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(`Failed to generate questions: ${response.status}`);
  }

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
    <form className="panel grid" onSubmit={handleSubmit}>
      <div>
        <h2>Start screening session</h2>
        <p className="muted">Use a resume file or pasted text to seed the interview.</p>
      </div>
      <div className="split">
        <div className="field">
          <label htmlFor="candidate-name">Candidate name</label>
          <input
            id="candidate-name"
            value={candidateName}
            onChange={(event) => setCandidateName(event.target.value)}
            placeholder="Enter candidate name"
          />
        </div>

        <div className="field">
          <label htmlFor="role">Role</label>
          <select id="role" value={role} onChange={(event) => setRole(event.target.value)}>
            <option value="backend engineer">Backend Engineer</option>
            <option value="ai ml engineer">AI / ML Engineer</option>
            <option value="frontend engineer">Frontend Engineer</option>
          </select>
        </div>
      </div>

      <div className="field">
        <label htmlFor="resume-file">Resume file</label>
        <input
          id="resume-file"
          type="file"
          accept=".pdf,.txt"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
        />
      </div>

      <div className="field">
        <label htmlFor="resume-text">Optional resume text</label>
        <textarea
          id="resume-text"
          value={resumeText}
          onChange={(event) => setResumeText(event.target.value)}
          placeholder="Paste a resume here if you do not want to upload a file."
        />
      </div>

      {error ? (
        <p className="muted">{error}</p>
      ) : (
        <p className="muted">Detected skills and retrieved context will be stored with the session.</p>
      )}

      <div className="actions">
        <button className="button" type="submit" disabled={loading}>
          {loading ? "Starting interview..." : "Start interview"}
        </button>
      </div>
    </form>
  );
}
