/** Mirrors the Python backend's JSON shape (snake_case, from app/schemas/*.py).
 * There's no shared-schema package across languages here — Pydantic and TS
 * can't share types directly — so these are kept in sync by hand. */

export type Source = "patient_reported" | "asked_and_denied" | "document_sourced" | "inferred" | "not_asked";

export interface Fact {
  id: string;
  field: string;
  value: string;
  source: Source;
  evidence_span: string | null;
  confidence: number;
  status: "unconfirmed" | "confirmed" | "corrected";
  timestamp: string;
  supersedes?: string | null;
  question_event_id?: string | null;
}

export interface IntakeRecord {
  session_id: string;
  protocol_id: string;
  facts: Fact[];
  created_at: string;
  updated_at: string;
}

export interface MissingFieldInfo {
  field: string;
  label: string;
  category: string;
}

export interface SafetyLogEntry {
  id: string;
  fact_id: string;
  triggered: boolean;
  rule_id: string | null;
  action: string | null;
  timestamp: string;
}

export interface AssistanceRequestInfo {
  id: string;
  reason: string;
  timestamp: string;
}

export interface DocumentMeta {
  id: string;
  filename: string;
  uploaded_at: string;
}
