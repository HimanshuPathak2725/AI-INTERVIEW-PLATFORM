import { useState } from "react";

import InterviewSession from "./components/InterviewSession";
import ResumeUpload from "./components/ResumeUpload";
import SummaryView from "./components/SummaryView";

function App() {
  const [session, setSession] = useState(null);
  const [summary, setSummary] = useState(null);

  const handleSessionCreated = (newSession) => {
    setSession(newSession);
    setSummary(null);
  };

  const handleComplete = (sessionSummary) => {
    setSummary(sessionSummary);
    setSession(null);
  };

  const handleRestart = () => {
    setSession(null);
    setSummary(null);
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div>
          <p className="eyebrow">AI Interview Platform</p>
          <h1>Candidate Screening</h1>
        </div>
        <ol className="stage-list">
          <li className={!session && !summary ? "active" : ""}>Profile</li>
          <li className={session ? "active" : ""}>Interview</li>
          <li className={summary ? "active" : ""}>Summary</li>
        </ol>
      </aside>
      <main className="content">
        {!session && !summary && <ResumeUpload onSessionCreated={handleSessionCreated} />}
        {session && !summary && <InterviewSession session={session} onComplete={handleComplete} />}
        {summary && <SummaryView summary={summary} onRestart={handleRestart} />}
      </main>
    </div>
  );
}

export default App;
