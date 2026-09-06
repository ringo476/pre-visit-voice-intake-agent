import type { TranscriptEntry } from "../hooks/useVoiceSession";

export function Transcript({ entries }: { entries: TranscriptEntry[] }) {
  return (
    <div className="transcript" role="log" aria-live="polite">
      {entries.length === 0 && <p className="transcript-empty">The conversation will appear here as you talk.</p>}
      {entries.map((entry) => (
        <div key={entry.id} className={`transcript-line transcript-line-${entry.speaker}`}>
          <span className="transcript-speaker">{entry.speaker === "patient" ? "You" : "Ava"}</span>
          <span className="transcript-text">{entry.text}</span>
        </div>
      ))}
    </div>
  );
}
