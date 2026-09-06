import { useVoiceSession } from "../hooks/useVoiceSession";
import { useIntakeState } from "../hooks/useIntakeState";
import { VoiceIndicator } from "./VoiceIndicator";
import { Transcript } from "./Transcript";
import { LiveBrief } from "./LiveBrief";

interface ConversationScreenProps {
  onFinish: () => void;
}

export function ConversationScreen({ onFinish }: ConversationScreenProps) {
  const { voiceState, transcript, errorMessage, endSession } = useVoiceSession();
  const intake = useIntakeState();

  function handleFinish() {
    endSession();
    onFinish();
  }

  return (
    <div className="screen conversation-screen">
      <div className="conversation-top">
        <VoiceIndicator voiceState={voiceState} />
        <button className="secondary-button" onClick={handleFinish}>
          I'm done
        </button>
      </div>

      {errorMessage && <p className="error-text">{errorMessage}</p>}

      <div className="conversation-panes">
        <div className="pane pane-transcript">
          <h2>Conversation</h2>
          <Transcript entries={transcript} />
        </div>
        <div className="pane pane-brief">
          <LiveBrief intake={intake} />
        </div>
      </div>
    </div>
  );
}
