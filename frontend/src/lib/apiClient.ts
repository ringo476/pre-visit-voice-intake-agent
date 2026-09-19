const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8080";

export interface MockAppointmentResponse {
  session_id: string;
  access_token: string;
}

/** Stands in for the patient having already received and opened a real
 * booking link (SMS/email) containing a signed access token — see
 * main.py's /api/appointments/mock for what's actually simulated here
 * versus what's genuinely verified server-side. Requires explicit consent;
 * the backend refuses to create a session at all without it. */
export async function bookMockAppointment(consent: boolean): Promise<MockAppointmentResponse> {
  const res = await fetch(`${API_BASE_URL}/api/appointments/mock`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ consent }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Failed to book appointment (${res.status})`);
  }
  return res.json();
}

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
