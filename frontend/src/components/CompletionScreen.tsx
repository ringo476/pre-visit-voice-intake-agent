import { useState } from "react";
import { useIntakeState } from "../hooks/useIntakeState";
import { getCurrentFactsMap } from "../lib/deriveCurrentFacts";
import { fetchClinicianBrief } from "../lib/apiClient";
import { ClinicianView } from "./ClinicianView";

export function CompletionScreen({ sessionId }: { sessionId: string }) {
  const intake = useIntakeState();
  const current = getCurrentFactsMap(intake.record);
  const [showClinicianView, setShowClinicianView] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [copyStatus, setCopyStatus] = useState<string | null>(null);

  async function handleCopy() {
    try {
      const { text } = await fetchClinicianBrief(sessionId);
      await navigator.clipboard.writeText(text);
      setCopyStatus("Copied to clipboard.");
    } catch {
      setCopyStatus("Could not copy — try the clinician view instead.");
    }
    setTimeout(() => setCopyStatus(null), 3000);
  }

  return (
    <div className="screen completion-screen">
      <h1>Here's what we captured</h1>

      <section className="patient-summary">
        <h2>Your summary</h2>
        {current.size === 0 ? (
          <p>We didn't capture any details this time.</p>
        ) : (
          <ul>
            {Array.from(current.entries())
              .filter(([, fact]) => fact.source === "patient_reported" || fact.source === "document_sourced")
              .map(([field, fact]) => (
                <li key={field}>
                  <strong>{field.replace(/_/g, " ")}:</strong> {fact.value}
                </li>
              ))}
          </ul>
        )}
      </section>

      {intake.missingFields.length > 0 && (
        <section className="patient-summary">
          <h2>Not yet covered</h2>
          <ul>
            {intake.missingFields.map((m) => (
              <li key={m.field}>{m.label}</li>
            ))}
          </ul>
        </section>
      )}

      <div className="completion-actions">
        <button className="primary-button" onClick={() => setSubmitted(true)} disabled={submitted}>
          {submitted ? "Submitted to clinic ✓" : "Submit to clinic"}
        </button>
        <button className="secondary-button" onClick={() => setShowClinicianView((v) => !v)}>
          {showClinicianView ? "Hide clinician view" : "Show clinician view"}
        </button>
        <button className="secondary-button" onClick={handleCopy}>
          Copy structured record
        </button>
      </div>
      {copyStatus && <p className="copy-status">{copyStatus}</p>}

      {showClinicianView && (
        <section className="clinician-view-panel">
          <h2>Clinician view</h2>
          <ClinicianView sessionId={sessionId} />
        </section>
      )}
    </div>
  );
}
