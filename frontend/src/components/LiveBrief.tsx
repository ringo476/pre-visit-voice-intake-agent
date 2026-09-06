import { useRef, useState } from "react";
import type { IntakeState } from "../hooks/useIntakeState";
import { useVoiceSession } from "../hooks/useVoiceSession";
import { getCurrentFactsMap } from "../lib/deriveCurrentFacts";

const SOURCE_LABEL: Record<string, string> = {
  patient_reported: "reported",
  asked_and_denied: "denied",
  document_sourced: "from document",
  inferred: "inferred",
  not_asked: "not asked",
};

export function LiveBrief({ intake }: { intake: IntakeState }) {
  const { uploadDocument } = useVoiceSession();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  const current = getCurrentFactsMap(intake.record);
  const triggeredSafety = intake.safetyLog.filter((e) => e.triggered);

  async function handleFileChosen(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setUploadStatus(`Uploading ${file.name}…`);
    try {
      await uploadDocument(file);
      setUploadStatus(`Attached ${file.name}`);
    } catch (err) {
      setUploadStatus(err instanceof Error ? err.message : "Upload failed");
    }
    setTimeout(() => setUploadStatus(null), 4000);
  }

  return (
    <div className="live-brief">
      <h2>Visit brief (live)</h2>
      <p className="protocol-status">
        {intake.protocolName ? `Focus: ${intake.protocolName}` : "Listening for what brings you in today…"}
      </p>

      {triggeredSafety.length > 0 && (
        <div className="safety-banner" role="alert">
          {triggeredSafety[triggeredSafety.length - 1].action === "emergency_escalation"
            ? "An emergency-level safety concern was flagged during this conversation. Please follow the guidance Ava gave you."
            : triggeredSafety[triggeredSafety.length - 1].action === "urgent_escalation"
              ? "A time-sensitive concern was flagged — Ava suggested being seen sooner than your scheduled appointment."
              : "A safety check was triggered."}
        </div>
      )}

      {current.size === 0 ? (
        <p className="live-brief-empty">Nothing captured yet — start talking and this will fill in.</p>
      ) : (
        <ul className="live-brief-facts">
          {Array.from(current.entries()).map(([field, fact]) => (
            <li key={field}>
              <span className="fact-field">{field.replace(/_/g, " ")}</span>
              <span className="fact-value">
                {fact.value} <span className="fact-source">({SOURCE_LABEL[fact.source] ?? fact.source})</span>
              </span>
            </li>
          ))}
        </ul>
      )}

      {intake.missingFields.length > 0 && (
        <>
          <h3>Still to cover</h3>
          <ul className="live-brief-missing">
            {intake.missingFields.map((m) => (
              <li key={m.field}>{m.label}</li>
            ))}
          </ul>
        </>
      )}

      <div className="document-upload">
        <h3>Documents</h3>
        {intake.documents.length > 0 && (
          <ul className="document-list">
            {intake.documents.map((d) => (
              <li key={d.id}>{d.filename}</li>
            ))}
          </ul>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,image/jpeg,image/png,image/webp"
          style={{ display: "none" }}
          onChange={handleFileChosen}
        />
        <button className="secondary-button" onClick={() => fileInputRef.current?.click()}>
          Attach a document
        </button>
        {uploadStatus && <p className="upload-status">{uploadStatus}</p>}
      </div>
    </div>
  );
}
