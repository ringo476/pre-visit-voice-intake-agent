import { useState } from "react";
import { useVoiceSession } from "../hooks/useVoiceSession";

interface WelcomeScreenProps {
  onStart: () => void;
}

export function WelcomeScreen({ onStart }: WelcomeScreenProps) {
  const { startSession, errorMessage } = useVoiceSession();
  const [starting, setStarting] = useState(false);

  async function handleStart() {
    setStarting(true);
    await startSession();
    setStarting(false);
    onStart();
  }

  return (
    <div className="screen welcome-screen">
      <div className="welcome-card">
        <div className="logo-dot" aria-hidden="true" />
        <p className="welcome-eyebrow">Continuing from your appointment confirmation</p>
        <h1>Prepare for your appointment</h1>
        <p className="welcome-copy">
          Tell us what's going on before your visit. Ava, our intake assistant, will ask a few questions and
          organize your story so your clinician can focus on helping you.
        </p>
        <p className="welcome-copy welcome-copy-muted">
          This is not a diagnosis and does not replace medical advice. If this is a medical emergency, call your
          local emergency number now.
        </p>

        <button className="primary-button" onClick={handleStart} disabled={starting}>
          {starting ? "Requesting microphone…" : "Start voice intake"}
        </button>

        {errorMessage && <p className="error-text">{errorMessage}</p>}

        <p className="welcome-copy-muted small">Synthetic demo patient · English</p>
      </div>
    </div>
  );
}
