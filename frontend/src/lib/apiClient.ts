const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8080";

export interface ClinicianBriefResponse {
  brief: {
    sections: { title: string; lines: string[] }[];
    clarifications: string[];
    disclaimer: string;
  };
  text: string;
}

export async function fetchClinicianBrief(sessionId: string): Promise<ClinicianBriefResponse> {
  const res = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/brief`);
  if (!res.ok) throw new Error(`Failed to fetch clinician brief (${res.status})`);
  return res.json();
}

export async function fetchFhirExport(sessionId: string): Promise<unknown> {
  const res = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/fhir`);
  if (!res.ok) throw new Error(`Failed to fetch FHIR export (${res.status})`);
  return res.json();
}

export async function uploadDocument(sessionId: string, file: File): Promise<{ document_id: string; filename: string; extraction_method: string }> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/documents`, { method: "POST", body: formData });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Failed to upload document (${res.status})`);
  }
  return res.json();
}
