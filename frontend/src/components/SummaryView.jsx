export default function SummaryView({ summary, onRestart }) {
  return (
    <section className="panel summary grid">
      <h2>Interview Summary</h2>

      <div className="split">
        <div className="field">
          <label>Role</label>
          <div className="muted">{summary.role}</div>
        </div>

        <div className="field">
          <label>Average score</label>
          <div className="muted">{summary.average_score ?? "N/A"}</div>
        </div>
      </div>

      <div className="split">
        <div className="field">
          <label>Strengths</label>
          <ul className="summary-list">
            {(summary.strengths || []).map((item) => (
              <li key={item} className="muted">
                {item}
              </li>
            ))}
          </ul>
        </div>

        <div className="field">
          <label>Gaps</label>
          <ul className="summary-list">
            {(summary.gaps || []).map((item) => (
              <li key={item} className="muted">
                {item}
              </li>
            ))}
          </ul>
        </div>
      </div>

      <p className="muted">{summary.overall_feedback}</p>
      <p className="muted">{summary.generated_report}</p>

      <div className="actions">
        <button className="button" type="button" onClick={onRestart}>
          Start another interview
        </button>
      </div>
    </section>
  );
}
