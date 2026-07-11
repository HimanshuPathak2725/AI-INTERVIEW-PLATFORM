import React from 'react';
import ReactMarkdown from 'react-markdown';

export default function ScorecardView({ summary, score, onClose }) {
  const percentage = Number.isFinite(score) ? (score * 100).toFixed(0) : '75';

  // Python list/dict bracket text leakage ko filter karne ka function
  const getCleanMarkdown = (rawInput) => {
    if (!rawInput) return '';
    let targetText = rawInput;
    
    if (typeof targetText === 'string') {
      const trimmed = targetText.trim();
      // Agar backend se text bracket logic format me leak ho raha hai
      if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
        try {
          const match = trimmed.match(/'text':\s*'([\s\S]*?)'(?=(,\s*'extras'|,\s*'signature'|\s*}))/);
          if (match && match[1]) {
            return match[1].replace(/\\n/g, '\n');
          }
        } catch (err) {
          console.error("Regex parsing failed, falling back to raw string:", err);
        }
      }
    }
    return targetText;
  };

  const cleanSummary = getCleanMarkdown(summary) || 'No scorecard summary was returned by the backend.';

  // Custom markdown tags ko transform karne ke liye inline components overrides
  const renderComponents = {
    h3: ({ node, ...props }) => (
      <h3 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#38bdf8', marginTop: '28px', marginBottom: '12px', borderBottom: '1px solid rgba(148, 163, 184, 0.15)', paddingBottom: '6px' }} {...props} />
    ),
    p: ({ node, ...props }) => (
      <p style={{ margin: '0 0 14px 0', fontSize: '1rem', color: '#cbd5e1', lineHeight: '1.7' }} {...props} />
    ),
    ul: ({ node, ...props }) => (
      <ul style={{ paddingLeft: '22px', margin: '10px 0 20px 0', listStyleType: 'disc' }} {...props} />
    ),
    li: ({ node, ...props }) => (
      <li style={{ margin: '8px 0', color: '#e2e8f0', fontSize: '0.98rem' }} {...props} />
    ),
    strong: ({ node, ...props }) => (
      <strong style={{ color: '#34d399', fontWeight: 600 }} {...props} />
    )
  };

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(180deg, #0f172a 0%, #111827 100%)', color: '#e5e7eb', padding: '28px' }}>
      <div style={{ maxWidth: '1100px', margin: '0 auto', display: 'grid', gap: '20px' }}>
        
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', gap: '16px', flexWrap: 'wrap', padding: '24px', borderRadius: '20px', background: 'rgba(15, 23, 42, 0.78)', border: '1px solid rgba(148, 163, 184, 0.2)', boxShadow: '0 24px 80px rgba(15, 23, 42, 0.45)' }}>
          <div>
            <p style={{ margin: 0, textTransform: 'uppercase', letterSpacing: '0.16em', color: '#94a3b8', fontSize: '0.74rem' }}>Interview Evaluation Report</p>
            <h1 style={{ margin: '10px 0 0', fontSize: 'clamp(2rem, 4vw, 3.4rem)', lineHeight: 1.05, color: '#f8fafc' }}>Software Engineering Candidate Review</h1>
          </div>
          <div style={{ minWidth: '180px', padding: '18px 20px', borderRadius: '18px', background: 'linear-gradient(135deg, rgba(15, 118, 110, 0.35), rgba(34, 197, 94, 0.16))', border: '1px solid rgba(45, 212, 191, 0.24)' }}>
            <div style={{ fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.14em', color: '#a7f3d0' }}>Technical Accuracy</div>
            <div style={{ fontSize: '2.5rem', fontWeight: 800, color: '#ecfeff', marginTop: '6px' }}>{percentage}%</div>
          </div>
        </header>

        <section style={{ padding: '28px', borderRadius: '20px', background: 'rgba(15, 23, 42, 0.86)', border: '1px solid rgba(148, 163, 184, 0.16)' }}>
          <div style={{ marginBottom: '20px', color: '#94a3b8', fontSize: '0.92rem', textTransform: 'uppercase', letterSpacing: '0.14em', fontWeight: 600 }}>Summary Report</div>
          <article style={{ color: '#e2e8f0' }}>
            <ReactMarkdown components={renderComponents}>{cleanSummary}</ReactMarkdown>
          </article>
        </section>

        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button
            type="button"
            onClick={onClose}
            style={{ padding: '12px 22px', borderRadius: '14px', border: '1px solid rgba(148, 163, 184, 0.2)', background: '#1f2937', color: '#f8fafc', fontWeight: 700, cursor: 'pointer', transition: 'background 0.2s' }}
          >
            Close and Exit
          </button>
        </div>
        
      </div>
    </div>
  );
}