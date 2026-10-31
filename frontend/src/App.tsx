import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const API_BASE = import.meta.env.DEV ? '' : 'http://127.0.0.1:8765';

type Message = {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  agent?: string;
  taskType?: string;
  sources?: string[];
};

type ChatResponse = {
  success: boolean;
  agent: string;
  task_type: string;
  reason: string;
  response: string;
  sources: string[];
};

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [backendOnline, setBackendOnline] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({
      behavior: 'smooth',
      block: 'end',
    });
  }, [messages, loading]);

  useEffect(() => {
    let active = true;

    const checkBackend = async () => {
      try {
        const response = await fetch(`${API_BASE}/health`);
        if (active) setBackendOnline(response.ok);
      } catch {
        if (active) setBackendOnline(false);
      }
    };

    checkBackend();
    const interval = window.setInterval(checkBackend, 10000);

    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  async function sendMessage() {
    const message = input.trim();

    if (!message || loading) return;

    setMessages((previous) => [
      ...previous,
      { id: Date.now(), role: 'user', content: message },
    ]);
    setInput('');

    if (textareaRef.current) {
      textareaRef.current.style.height = '32px';
      textareaRef.current.style.overflowY = 'hidden';
    }

    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      });

      if (!response.ok) {
        const error = await response.json().catch(() => null);
        throw new Error(error?.detail || `Request failed: ${response.status}`);
      }

      const result: ChatResponse = await response.json();

      setMessages((previous) => [
        ...previous,
        {
          id: Date.now() + 1,
          role: 'assistant',
          content: result.response,
          agent: result.agent,
          taskType: result.task_type,
          sources: result.sources,
        },
      ]);
    } catch (error) {
      setMessages((previous) => [
        ...previous,
        {
          id: Date.now() + 1,
          role: 'assistant',
          content:
            error instanceof Error
              ? `Unable to reach SALAZAR: ${error.message}`
              : 'An unexpected error occurred.',
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">S</div>
          <div>
            <div className="brand-name">SALAZAR</div>
            <div className="brand-caption">OFFLINE AI WORKSTATION</div>
          </div>
        </div>

        <button className="new-chat" onClick={() => setMessages([])}>
          <span>＋</span> New conversation
        </button>

        <div className="sidebar-section">
          <div className="section-label">WORKSPACE</div>
          <button className="nav-item active">
            <span>◈</span> Chat
          </button>
          <button className="nav-item">
            <span>▤</span> Documents
          </button>
          <button className="nav-item">
            <span>⌘</span> Workflows
          </button>
          <button className="nav-item">
            <span>▧</span> Models
          </button>
        </div>

        <div className="sidebar-section history-section">
          <div className="section-label">RECENT</div>
          {messages.length > 0 ? (
            <div className="history-entry">Current conversation</div>
          ) : (
            <div className="empty-history">No conversations yet</div>
          )}
        </div>

        <div className="sidebar-bottom">
          <button className="nav-item">
            <span>⚙</span> Settings
          </button>
          <div className="profile">
            <div className="profile-avatar">S</div>
            <div>
              <div className="profile-name">SALAZAR Core</div>
              <div className="profile-caption">Local environment</div>
            </div>
            <span className="profile-menu">···</span>
          </div>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Workspace</span>
            <span className="breadcrumb-divider">/</span>
            <strong>Chat</strong>
          </div>

          <div className="topbar-right">
            <div className="connection">
              <span className={`status-dot ${backendOnline ? 'online' : ''}`} />
              {backendOnline ? 'Backend connected' : 'Backend offline'}
            </div>
            <div className="local-badge">
              <span>◈</span> LOCAL MODE
            </div>
          </div>
        </header>

        <section className="chat-area">
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-symbol">S</div>
              <div className="eyebrow">YOUR PRIVATE AI WORKSPACE</div>
              <h1>Intelligence, <span>without boundaries.</span></h1>
              <p>
                Your models. Your documents. Your machine.
                <br />
                Everything runs locally, under your control.
              </p>

              <div className="suggestions">
                <button onClick={() => setInput('Explain how a local language model works.')}>
                  <span>◈</span>
                  <div>
                    <strong>Explore an idea</strong>
                    <small>Ask your local model anything</small>
                  </div>
                  <span className="suggestion-arrow">↗</span>
                </button>
                <button onClick={() => setInput('Write a Python function to reverse a linked list.')}>
                  <span>⌘</span>
                  <div>
                    <strong>Build something</strong>
                    <small>Get help with code and debugging</small>
                  </div>
                  <span className="suggestion-arrow">↗</span>
                </button>
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message) => (
                <article
                  className={`message ${message.role}`}
                  key={message.id}
                >
                  <div className="message-avatar">
                    {message.role === 'user' ? 'A' : 'S'}
                  </div>
                  <div className="message-body">
                    <div className="message-heading">
                      <strong>
                        {message.role === 'user' ? 'You' : 'SALAZAR'}
                      </strong>
                      {message.agent && (
                        <span className="agent-tag">
                          {message.agent} · {message.taskType}
                        </span>
                      )}
                    </div>
                    <div className="message-content">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {message.content}
                      </ReactMarkdown>
                    </div>
                    {message.sources && message.sources.length > 0 && (
                      <div className="sources">
                        Sources: {message.sources.join(', ')}
                      </div>
                    )}
                  </div>
                </article>
              ))}
              {loading && (
                <div className="thinking">
                  <span className="thinking-dot" />
                  SALAZAR is thinking...
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
          )}
        </section>

        <footer className="composer-area">
          <div className="composer">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(event) => {
                setInput(event.target.value);

                const textarea = event.currentTarget;
                textarea.style.height = 'auto';

                const maxHeight = 220;
                textarea.style.height = `${Math.min(textarea.scrollHeight, maxHeight)}px`;
                textarea.style.overflowY =
                  textarea.scrollHeight > maxHeight ? 'auto' : 'hidden';
              }}
              onKeyDown={handleKeyDown}
              placeholder="Message SALAZAR..."
              rows={1}
              disabled={loading}
            />
            <div className="composer-bottom">
              <div className="composer-hint">
                <span>◈</span> Local inference
                <span className="hint-divider">·</span>
                Enter to send
                <span className="hint-divider">·</span>
                Shift + Enter for new line
              </div>
              <button
                className="send-button"
                onClick={() => void sendMessage()}
                disabled={!input.trim() || loading}
                aria-label="Send message"
              >
                ↑
              </button>
            </div>
          </div>
          <div className="disclaimer">
            SALAZAR may generate inaccurate information. Verify important outputs.
          </div>
        </footer>
      </main>
    </div>
  );
}

export default App;
