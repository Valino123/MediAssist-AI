import { useRef, useState, type FormEvent } from 'react';

type Sender = 'user' | 'assistant';

type Message = {
  id: string;
  sender: Sender;
  content: string;
  agentTag?: string;
  imageUrl?: string;
};

const quickPrompts = [
  'What are the symptoms of COVID-19?',
  'Help me search the symptoms of rhinitis.',
  'Analyze this medical image.',
  "Let's talk about recent research on cancer.",
];

const apiBase = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '');
const buildUrl = (path: string) => `${apiBase}${path}`;

const uid = () => (crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2));

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [searchType, setSearchType] = useState<'general' | 'literature'>('general');
  const [selectedImage, setSelectedImage] = useState<{ file: File; preview: string } | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [recording, setRecording] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const hasMessages = messages.length > 0;

  const addMessage = (message: Message) => {
    setMessages((prev) => [...prev, message]);
  };

  const handlePromptClick = (prompt: string) => {
    setInput(prompt);
  };

  const handleImageChange = (file?: File) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const preview = (event.target?.result as string) || '';
      setSelectedImage({ file, preview });
    };
    reader.readAsDataURL(file);
  };

  const removeImage = () => {
    setSelectedImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const sendRequest = async (messageText: string, image?: { file: File }) => {
    if (image) {
      const formData = new FormData();
      formData.append('query', messageText || 'Please analyze this medical image');
      formData.append('image', image.file);
      return fetch(buildUrl('/analyze-image'), {
        method: 'POST',
        body: formData,
      });
    }

    const params = new URLSearchParams();
    if (searchType) params.set('searchType', searchType);

    return fetch(buildUrl(`/chat${params.toString() ? `?${params}` : ''}`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: messageText }),
    });
  };

  const handleSubmit = async (event?: FormEvent) => {
    event?.preventDefault();
    if (isSending) return;
    const trimmed = input.trim();
    if (!trimmed && !selectedImage) return;

    const userMsg: Message = {
      id: uid(),
      sender: 'user',
      content: trimmed || 'Analyzing image...',
      imageUrl: selectedImage?.preview,
    };
    addMessage(userMsg);
    setIsSending(true);

    try {
      const response = await sendRequest(trimmed, selectedImage ?? undefined);
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        addMessage({
          id: uid(),
          sender: 'assistant',
          content: `Error ${response.status}: ${err.detail || err.error || 'Unknown error'}`,
          agentTag: 'Error',
        });
        return;
      }
      const data = await response.json();
      addMessage({
        id: uid(),
        sender: 'assistant',
        content: data.response || data.message || data.text || 'No response received.',
        agentTag: data.agent || data.status || 'MEDIASSIST',
      });
    } catch (error) {
      addMessage({
        id: uid(),
        sender: 'assistant',
        content: 'Sorry, there was an error processing your request.',
        agentTag: 'Error',
      });
    } finally {
      setIsSending(false);
      setInput('');
      removeImage();
    }
  };

  const blobToBase64 = (blob: Blob) =>
    new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve((reader.result as string).split(',')[1]);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });

  const startRecording = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      addMessage({
        id: uid(),
        sender: 'assistant',
        content: 'Microphone not available in this browser.',
        agentTag: 'VOICE_AGENT',
      });
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      audioChunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      recorder.onstop = async () => {
        try {
          const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          const base64 = await blobToBase64(blob);
          const resp = await fetch(buildUrl('/voice/stt'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ audio_base64: `data:audio/webm;base64,${base64}` }),
          });
          const data = await resp.json();
          if (data.status === 'success' && data.text) {
            setInput(data.text);
          } else {
            addMessage({
              id: uid(),
              sender: 'assistant',
              content: data.error || 'Speech-to-text failed.',
              agentTag: 'VOICE_AGENT',
            });
          }
        } catch (err) {
          addMessage({
            id: uid(),
            sender: 'assistant',
            content: 'Voice capture failed.',
            agentTag: 'VOICE_AGENT',
          });
        }
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      setRecording(true);
    } catch (error) {
      addMessage({
        id: uid(),
        sender: 'assistant',
        content: 'Microphone permission denied.',
        agentTag: 'VOICE_AGENT',
      });
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
    }
    setRecording(false);
  };

  const handleVoiceClick = () => {
    if (recording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const speak = async (text: string) => {
    try {
      const resp = await fetch(buildUrl('/voice/tts'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ text }),
      });
      const data = await resp.json();
      if (data.status === 'success' && data.audio_data) {
        const audio = new Audio(data.audio_data);
        audio.play();
      }
    } catch {
      // best-effort
    }
  };

  return (
    <div className="app-canvas">
      <div className="chat-container">
        <header className="chat-header">
          <div className="header-left">
            <div className="logo-container">
              <div className="logo-icon">🩺</div>
            </div>
            <div>
              <h3 className="mb-1 fw-bold text-primary">MediAssist</h3>
              <small className="text-muted">
                AI-powered multi-agent system for medical diagnosis and assistance
              </small>
            </div>
          </div>
          <div className="header-right">
            <div className="search-type-selector">
              <label htmlFor="searchTypeSelect" className="form-label">
                Search Mode:
              </label>
              <select
                id="searchTypeSelect"
                className="modern-select"
                value={searchType}
                onChange={(e) => setSearchType(e.target.value as 'general' | 'literature')}
              >
                <option value="general">General Web</option>
                <option value="literature">PubMed Literature</option>
              </select>
            </div>
            <div className="status-indicator">
              <div className="status-dot online" />
              <span className="status-text">Online</span>
            </div>
          </div>
        </header>

        <main className="chat-body" id="chat-body">
          {!hasMessages && (
            <div className="welcome-message">
              <div className="welcome-icon">❤️</div>
              <h2 className="welcome-title">Welcome to MediAssist</h2>
              <p className="welcome-subtitle">
                Your AI-powered medical assistant is ready to help with health-related questions and
                medical image analysis.
              </p>
              <p className="welcome-helper">
                Choose a prompt below or write your own to start chatting with MediAssist AI
              </p>
              <div className="prompt-chips">
                {quickPrompts.map((prompt) => (
                  <button key={prompt} type="button" className="chip" onClick={() => handlePromptClick(prompt)}>
                    {prompt}
                  </button>
                ))}
              </div>
              <form
                id="hero-chat-form"
                className="hero-form"
                onSubmit={(e) => {
                  e.preventDefault();
                  handleSubmit();
                }}
              >
                <div className="hero-input-group">
                  <input
                    type="text"
                    id="heroMessageInput"
                    className="hero-input"
                    placeholder="How can MediAssist help you today?"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                  />
                  <button type="submit" className="hero-send" title="Send">
                    ➤
                  </button>
                </div>
              </form>
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className={`message ${message.sender}-message`}>
              {message.agentTag && <div className="agent-tag">{message.agentTag}</div>}
              <div className="message-content">
                {message.content}
                {message.sender === 'assistant' && (
                  <button className="tts-btn" title="Play audio" onClick={() => speak(message.content)}>
                    🔊
                  </button>
                )}
              </div>
              {message.imageUrl && (
                <img src={message.imageUrl} className="img-preview" alt="uploaded" />
              )}
            </div>
          ))}

          {isSending && (
            <div className="message assistant-message thinking">
              <div className="dot" />
              <div className="dot" />
              <div className="dot" />
            </div>
          )}
        </main>

        {hasMessages && (
          <footer className="chat-footer">
            <form id="chat-form" onSubmit={handleSubmit}>
              <div className="input-row">
                <button
                  type="button"
                  className={`icon-button ${recording ? 'active' : ''}`}
                  onClick={handleVoiceClick}
                  title={recording ? 'Stop recording' : 'Voice input'}
                >
                  🎤
                </button>

                <button
                  type="button"
                  className="icon-button"
                  onClick={() => fileInputRef.current?.click()}
                  title="Upload image"
                >
                  🖼️
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  style={{ display: 'none' }}
                  onChange={(e) => handleImageChange(e.target.files?.[0])}
                />

                <input
                  type="text"
                  className="text-input"
                  placeholder="Ask me anything about health or upload a medical image..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  required={!selectedImage}
                />

                <button type="submit" className="send-button" disabled={isSending}>
                  Send
                </button>
              </div>
            </form>

            {selectedImage && (
              <div className="image-preview">
                <img src={selectedImage.preview} className="img-preview" alt="preview" />
                <div className="image-preview-meta">
                  <small>Image ready for analysis</small>
                  <button type="button" className="remove-image" onClick={removeImage}>
                    Remove
                  </button>
                </div>
              </div>
            )}

            <div className="footer-info">
              <small>
                <span role="img" aria-label="shield">
                  🛡️
                </span>{' '}
                Your conversations are secure and private
              </small>
            </div>
          </footer>
        )}
      </div>
    </div>
  );
}

export default App;
