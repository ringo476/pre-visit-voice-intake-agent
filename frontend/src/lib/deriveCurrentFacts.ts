import type { Fact, IntakeRecord } from "../types";

/**
 * Presentation-only mirror of the backend's get_current_facts
 * (app/state/state_engine.py): the latest non-superseded fact per field.
 * Kept local rather than shared across languages since Pydantic/TS can't
 * share a package — this read-only slice is small enough not to be worth
 * anything fancier for an MVP demo.
 */
export function getCurrentFactsMap(record: IntakeRecord | null): Map<string, Fact> {
  const map = new Map<string, Fact>();
  if (!record) return map;

  const superseded = new Set(record.facts.map((f) => f.supersedes).filter((id): id is string => !!id));
  for (const fact of record.facts) {
    if (superseded.has(fact.id)) continue;
    const existing = map.get(fact.field);
    if (!existing || fact.timestamp > existing.timestamp) map.set(fact.field, fact);
  }
  return map;
}
