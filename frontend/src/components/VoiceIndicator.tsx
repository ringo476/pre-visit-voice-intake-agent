import type { VoiceState } from "../hooks/useVoiceSession";

const STATE_LABEL: Record<VoiceState, string> = {
  idle: "Not connected",
  connecting: "Connecting…",
  listening: "Listening",
  thinking: "Thinking",
  speaking: "Speaking",
  error: "Connection issue",
};

export function VoiceIndicator({ voiceState }: { voiceState: VoiceState }) {
  return (
    <div className={`voice-indicator voice-indicator-${voiceState}`}>
      <div className="voice-orb" aria-hidden="true">
        <div className="voice-orb-core" />
      </div>
      <span className="voice-state-label">{STATE_LABEL[voiceState]}</span>
    </div>
  );
}
