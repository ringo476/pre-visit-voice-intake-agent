import { useEffect, useState } from "react";
import { fetchClinicianBrief, fetchFhirExport, type ClinicianBriefResponse } from "../lib/apiClient";

export function ClinicianView({ sessionId }: { sessionId: string }) {
  const [brief, setBrief] = useState<ClinicianBriefResponse | null>(null);
  const [fhir, setFhir] = useState<unknown | null>(null);
  const [showFhir, setShowFhir] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    fetchClinicianBrief(sessionId)
      .then(setBrief)
      .catch((err) => setLoadError(err instanceof Error ? err.message : String(err)));
  }, [sessionId]);

  async function toggleFhir() {
    if (!showFhir && !fhir) {
      try {
        setFhir(await fetchFhirExport(sessionId));
      } catch (err) {
        setLoadError(err instanceof Error ? err.message : String(err));
        return;
      }
    }
    setShowFhir((v) => !v);
  }

  if (loadError) return <p className="error-text">Could not load the clinician brief: {loadError}</p>;
  if (!brief) return <p>Loading clinician brief…</p>;

  return (
    <div className="clinician-view">
      <p className="disclaimer-banner">{brief.brief.disclaimer}</p>

      {brief.brief.sections.map((section) => (
        <div key={section.title} className="brief-section">
          <h3>{section.title}</h3>
          <p>{section.lines.join(" ")}</p>
        </div>
      ))}

      {brief.brief.clarifications.length > 0 && (
        <div className="brief-section brief-clarifications">
          <h3>Information requiring clarification</h3>
          <ul>
            {brief.brief.clarifications.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        </div>
      )}

      <button className="secondary-button" onClick={toggleFhir}>
        {showFhir ? "Hide FHIR export" : "Show FHIR export (demo)"}
      </button>

      {showFhir && fhir !== null && <pre className="fhir-json">{JSON.stringify(fhir, null, 2)}</pre>}
    </div>
  );
}
