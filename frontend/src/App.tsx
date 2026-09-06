import { useState } from "react";
import { VoiceSessionProvider, useVoiceSession } from "./hooks/useVoiceSession";
import { WelcomeScreen } from "./components/WelcomeScreen";
import { ConversationScreen } from "./components/ConversationScreen";
import { CompletionScreen } from "./components/CompletionScreen";

type Screen = "welcome" | "conversation" | "completion";

function AppShell() {
  const [screen, setScreen] = useState<Screen>("welcome");
  const { sessionId } = useVoiceSession();

  if (screen === "welcome") {
    return <WelcomeScreen onStart={() => setScreen("conversation")} />;
  }
  if (screen === "conversation") {
    return <ConversationScreen onFinish={() => setScreen("completion")} />;
  }
  if (!sessionId) {
    return <p className="screen">Finishing up…</p>;
  }
  return <CompletionScreen sessionId={sessionId} />;
}

export default function App() {
  return (
    <VoiceSessionProvider>
      <AppShell />
    </VoiceSessionProvider>
  );
}
