import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { uploadDocument as uploadDocumentRequest } from "../lib/apiClient";
import type { AssistanceRequestInfo, DocumentMeta, IntakeRecord, MissingFieldInfo, SafetyLogEntry } from "../types";

export type VoiceState = "idle" | "connecting" | "listening" | "thinking" | "speaking" | "error";

export interface TranscriptEntry {
  id: string;
  speaker: "patient" | "agent";
  text: string;
}

export interface IntakeState {
  record: IntakeRecord | null;
  // Null until the patient describes an actual complaint — the backend
  // classifies which protocol applies from what they say, rather than the
  // patient picking a category up front.
  protocolName: string | null;
  missingFields: MissingFieldInfo[];
  safetyLog: SafetyLogEntry[];
  assistanceRequests: AssistanceRequestInfo[];
  documents: DocumentMeta[];
  briefFinalized: boolean;
}

interface VoiceSessionContextValue {
  voiceState: VoiceState;
  sessionId: string | null;
  transcript: TranscriptEntry[];
  intake: IntakeState;
  errorMessage: string | null;
  startSession: () => Promise<void>;
  endSession: () => void;
  uploadDocument: (file: File) => Promise<void>;
}

const VoiceSessionContext = createContext<VoiceSessionContextValue | null>(null);

const WS_URL = (import.meta.env.VITE_WS_URL as string | undefined) ?? "ws://localhost:8080/ws";

// Tuned empirically in a real deployment; a fixed energy threshold is a
// reasonable MVP stand-in for a proper VAD model.
const SPEECH_ENERGY_THRESHOLD = 0.06;
const SILENCE_MS = 2500;

const emptyIntake: IntakeState = {
  record: null,
  protocolName: null,
  missingFields: [],
  safetyLog: [],
  assistanceRequests: [],
  documents: [],
  briefFinalized: false,
};

export function VoiceSessionProvider({ children }: { children: ReactNode }) {
  const [voiceState, setVoiceState] = useState<VoiceState>("idle");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [intake, setIntake] = useState<IntakeState>(emptyIntake);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const silenceTimerRef = useRef<number | null>(null);
  const isRecordingRef = useRef(false);
  const isPlayingRef = useRef(false);
  // True from the moment a recorded utterance is sent until the backend's
  // reply starts arriving — blocks the VAD loop from starting another
  // recording while a turn is already in flight, so repeating yourself
  // while the agent is "thinking" doesn't queue up several separate turns.
  const turnInFlightRef = useRef(false);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);
  const vadRafRef = useRef<number | null>(null);

  function stopPlayback() {
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current.currentTime = 0;
      currentAudioRef.current = null;
    }
    isPlayingRef.current = false;
  }

  function startRecording() {
    if (isRecordingRef.current || !micStreamRef.current) return;
    isRecordingRef.current = true;
    recordedChunksRef.current = [];

    const recorder = new MediaRecorder(micStreamRef.current, { mimeType: "audio/webm;codecs=opus" });
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunksRef.current.push(e.data);
    };
    recorder.onstop = () => {
      isRecordingRef.current = false;
      const blob = new Blob(recordedChunksRef.current, { type: "audio/webm" });
      recordedChunksRef.current = [];
      console.log("[VAD] recorder stopped, blob size:", blob.size, "ws readyState:", wsRef.current?.readyState);
      if (blob.size > 0) {
        turnInFlightRef.current = true;
        blob.arrayBuffer().then((buf) => wsRef.current?.send(buf));
      }
    };
    recorder.start();
    recorderRef.current = recorder;
    setVoiceState("listening");
  }

  function stopRecording() {
    console.log("[VAD] stopRecording() called, recorder state:", recorderRef.current?.state);
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
  }

  function scheduleSilenceStop() {
    if (silenceTimerRef.current) return; // already counting down, don't restart the clock
    console.log("[VAD] silence timer scheduled, will stop in 900ms unless speech resumes");
    silenceTimerRef.current = window.setTimeout(() => {
      silenceTimerRef.current = null;
      stopRecording();
    }, SILENCE_MS);
  }

  function vadLoop() {
    const analyser = analyserRef.current;
    if (!analyser) {
      console.log("[VAD] loop is running but analyserRef is null");
    }
    if (analyser) {
      const data = new Uint8Array(analyser.fftSize);
      analyser.getByteTimeDomainData(data);
      let sumSquares = 0;
      for (let i = 0; i < data.length; i++) {
        const norm = (data[i] - 128) / 128;
        sumSquares += norm * norm;
      }
      const rms = Math.sqrt(sumSquares / data.length);
      const speaking = rms > SPEECH_ENERGY_THRESHOLD;
      console.log("[VAD]", { rms: rms.toFixed(4), speaking, recording: isRecordingRef.current });

      if (speaking) {
        if (isPlayingRef.current) {
          // Barge-in: the patient started talking while the agent's reply was still playing.
          stopPlayback();
          wsRef.current?.send(JSON.stringify({ type: "barge_in" }));
          startRecording();
        } else if (!isRecordingRef.current && !turnInFlightRef.current) {
          startRecording();
        }
        if (silenceTimerRef.current) {
          window.clearTimeout(silenceTimerRef.current);
          silenceTimerRef.current = null;
        }
      } else if (isRecordingRef.current) {
        scheduleSilenceStop();
      }
    }
  }

  function playAudio(data: ArrayBuffer) {
    stopPlayback();
    const blob = new Blob([data], { type: "audio/mp3" });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    currentAudioRef.current = audio;
    isPlayingRef.current = true;
    audio.onended = () => {
      isPlayingRef.current = false;
      URL.revokeObjectURL(url);
      setVoiceState("listening");
    };
    audio.play().catch(() => {
      isPlayingRef.current = false;
    });
  }

  function handleControlMessage(msg: Record<string, unknown>) {
    switch (msg.type) {
      case "session_started":
        sessionIdRef.current = msg.session_id as string;
        setSessionId(msg.session_id as string);
        break;
      case "state_delta":
        setIntake({
          record: msg.record as IntakeRecord,
          protocolName: (msg.protocol_name as string | null) ?? null,
          missingFields: msg.missing_fields as MissingFieldInfo[],
          safetyLog: msg.safety_log as SafetyLogEntry[],
          assistanceRequests: msg.assistance_requests as AssistanceRequestInfo[],
          documents: msg.documents as DocumentMeta[],
          briefFinalized: msg.brief_finalized as boolean,
        });
        break;
      case "transcript":
        setTranscript((prev) => [
          ...prev,
          { id: `${prev.length}-${msg.speaker as string}`, speaker: msg.speaker as "patient" | "agent", text: msg.text as string },
        ]);
        break;
      case "voice_state":
        // "speaking" (a reply is on its way) and "listening" (e.g. after an
        // error, per main.py's error-recovery path) both mean this turn is
        // over — safe to accept a new recording again.
        if (msg.state === "speaking" || msg.state === "listening") {
          turnInFlightRef.current = false;
        }
        setVoiceState(msg.state as VoiceState);
        break;
      case "error":
        setErrorMessage(msg.message as string);
        break;
    }
  }

  async function startSession() {
    setErrorMessage(null);
    setVoiceState("connecting");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micStreamRef.current = stream;

      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;
      if (audioContext.state === "suspended") {
        await audioContext.resume();
      }
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      analyserRef.current = analyser;
    } catch {
      setErrorMessage("Microphone access is required to start the voice intake.");
      setVoiceState("error");
      return;
    }

    const ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => {
      // The backend speaks first (the opening line) the instant this
      // connection opens — that first turn is already "in flight" from
      // this exact moment, before any recording of ours triggers the usual
      // lock, so lock the mic here too until it actually starts speaking.
      turnInFlightRef.current = true;
      setVoiceState("listening");
      vadRafRef.current = window.setInterval(vadLoop, 100);
    };
    ws.onmessage = (event) => {
      if (typeof event.data === "string") {
        try {
          handleControlMessage(JSON.parse(event.data));
        } catch {
          // ignore malformed messages
        }
      } else {
        playAudio(event.data as ArrayBuffer);
      }
    };
    ws.onerror = () => {
      setErrorMessage("Connection to the intake service failed. Is the backend running?");
      setVoiceState("error");
    };
    ws.onclose = () => {
      // The connection can drop for reasons other than the user clicking
      // "I'm done" (backend restart, network blip) — clean up fully here
      // too, otherwise any in-progress audio/recording/VAD loop just keeps
      // running orphaned after the UI has already reset to the welcome screen.
      cleanupLocalMedia();
      setVoiceState((prev) => (prev === "error" ? prev : "idle"));
    };
  }

  function cleanupLocalMedia() {
    if (vadRafRef.current) window.clearInterval(vadRafRef.current);
    stopPlayback();
    stopRecording();
    micStreamRef.current?.getTracks().forEach((t) => t.stop());
    audioContextRef.current?.close().catch(() => {});
  }

  function endSession() {
    cleanupLocalMedia();
    wsRef.current?.close();
    setVoiceState("idle");
  }

  async function uploadDocument(file: File) {
    if (!sessionIdRef.current) throw new Error("No active session to attach a document to.");
    await uploadDocumentRequest(sessionIdRef.current, file);
    // The resulting agent reaction (transcript + state_delta + audio) arrives over the same WebSocket.
  }

  useEffect(() => {
    return () => endSession();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <VoiceSessionContext.Provider
      value={{ voiceState, sessionId, transcript, intake, errorMessage, startSession, endSession, uploadDocument }}
    >
      {children}
    </VoiceSessionContext.Provider>
  );
}

export function useVoiceSession(): VoiceSessionContextValue {
  const ctx = useContext(VoiceSessionContext);
  if (!ctx) throw new Error("useVoiceSession must be used within a VoiceSessionProvider");
  return ctx;
}

/** Thin selector over the same underlying connection — the intake-state slice only. */
export function useIntakeState(): IntakeState {
  return useVoiceSession().intake;
}
