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

type Conversation = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
};

type ChatResponse = {
  success: boolean;
  conversation_id: string;
  agent: string;
  task_type: string;
  reason: string;
  response: string;
  sources: string[];
};

type DocumentItem = {
  filename: string;
  size_bytes: number;
  chunks: number;
  indexed: boolean;
};

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [backendOnline, setBackendOnline] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversation, setActiveConversation] = useState<string | null>(null);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState('');
  const [activeView, setActiveView] = useState<'chat' | 'documents' | 'workflows' | 'models'>('chat');
  const [workflowTopic, setWorkflowTopic] = useState('');
  const [workflowCode, setWorkflowCode] = useState('');
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [documentError, setDocumentError] = useState('');
  const [documentMessage, setDocumentMessage] = useState('');
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  async function refreshDocuments() {
    setDocumentsLoading(true);
    setDocumentError('');

    try {
      const response = await fetch(`${API_BASE}/api/documents`);
      if (!response.ok) {
        const error = await response.json().catch(() => null);
        throw new Error(error?.detail || 'Unable to load documents.');
      }
      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (error) {
      setDocumentError(
        error instanceof Error ? error.message : 'Unable to load documents.'
      );
    } finally {
      setDocumentsLoading(false);
    }
  }

  async function uploadDocument(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setDocumentError('');
    setDocumentMessage('');

    try {
      const form = new FormData();
      form.append('file', file);

      const response = await fetch(`${API_BASE}/api/documents`, {
        method: 'POST',
        body: form,
      });

      const result = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(result?.detail || `Upload failed: ${response.status}`);
      }

      setDocumentMessage(
        `${result.filename} indexed successfully · ${result.chunks} chunks`
      );
      await refreshDocuments();
    } catch (error) {
      setDocumentError(
        error instanceof Error ? error.message : 'Document upload failed.'
      );
    } finally {
      setUploading(false);
      event.target.value = '';
    }
  }

  async function removeDocument(filename: string) {
    if (!window.confirm(`Delete "${filename}" and its indexed content?`)) return;

    setDocumentError('');
    setDocumentMessage('');

    try {
      const response = await fetch(
        `${API_BASE}/api/documents/${encodeURIComponent(filename)}`,
        { method: 'DELETE' }
      );
      const result = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(result?.detail || 'Unable to delete document.');
      }

      setDocumentMessage(`${filename} deleted successfully.`);
      await refreshDocuments();
    } catch (error) {
      setDocumentError(
        error instanceof Error ? error.message : 'Document deletion failed.'
      );
    }
  }

  function openDocuments() {
    setActiveView('documents');
    setDocumentMessage('');
    void refreshDocuments();
  }

  async function refreshConversations() {
    try {
      const response = await fetch(`${API_BASE}/api/conversations`);
      if (!response.ok) throw new Error('Unable to load conversation history.');
      const data = await response.json();
      setConversations(data.conversations || []);
      setHistoryError('');
      return data.conversations || [];
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : 'History unavailable.');
      return [];
    }
  }

  async function openConversation(id: string) {
    if (loading) return;

    try {
      setHistoryLoading(true);
      const response = await fetch(
        `${API_BASE}/api/conversations/${encodeURIComponent(id)}/messages`
      );
      if (!response.ok) throw new Error('Unable to open this conversation.');

      const data = await response.json();
      setMessages(
        (data.messages || []).map((message: any) => ({
          id: message.id,
          role: message.role,
          content: message.content,
          agent: message.agent || undefined,
          taskType: message.task_type || undefined,
          sources: message.sources || [],
        }))
      );
      setActiveConversation(id);
      setHistoryError('');
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : 'Unable to open conversation.');
    } finally {
      setHistoryLoading(false);
    }
  }

  function newConversation() {
    if (loading) return;
    setMessages([]);
    setActiveConversation(null);
    setInput('');
    setHistoryError('');
    textareaRef.current?.focus();
  }

  async function deleteConversation(id: string) {
    if (loading) return;
    if (!window.confirm('Delete this conversation permanently?')) return;

    try {
      const response = await fetch(
        `${API_BASE}/api/conversations/${encodeURIComponent(id)}`,
        { method: 'DELETE' }
      );
      if (!response.ok) throw new Error('Unable to delete conversation.');

      if (activeConversation === id) {
        setMessages([]);
        setActiveConversation(null);
      }
      await refreshConversations();
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : 'Delete failed.');
    }
  }

  useEffect(() => {
    let active = true;

    async function initialize() {
      const checkBackend = async () => {
        try {
          const response = await fetch(`${API_BASE}/health`);
          if (active) setBackendOnline(response.ok);
          return response.ok;
        } catch {
          if (active) setBackendOnline(false);
          return false;
        }
      };

      const online = await checkBackend();

      if (online && active) {
        const history = await refreshConversations();
        if (active && history.length > 0) {
          await openConversation(history[0].id);
        }
      }

      if (active) setHistoryLoading(false);
    }

    void initialize();
    const interval = window.setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE}/health`);
        if (active) setBackendOnline(response.ok);
      } catch {
        if (active) setBackendOnline(false);
      }
    }, 10000);

    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({
      behavior: 'smooth',
      block: 'end',
    });
  }, [messages, loading]);

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
        body: JSON.stringify({
          message,
          conversation_id: activeConversation,
        }),
      });

      if (!response.ok) {
        const error = await response.json().catch(() => null);
        throw new Error(error?.detail || `Request failed: ${response.status}`);
      }

      const result: ChatResponse = await response.json();

      setActiveConversation(result.conversation_id);
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

      await refreshConversations();
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

        <button className="new-chat" onClick={newConversation} disabled={loading}>
          <span>＋</span> New conversation
        </button>

        <div className="sidebar-section">
          <div className="section-label">WORKSPACE</div>
          <button
            className={`nav-item ${activeView === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveView('chat')}
          >
            <span>◈</span> Chat
          </button>
          <button
            className={`nav-item ${activeView === 'documents' ? 'active' : ''}`}
            onClick={openDocuments}
          >
            <span>▤</span> Documents
          </button>
          <button
            className={`nav-item ${activeView === 'workflows' ? 'active' : ''}`}
            onClick={() => setActiveView('workflows')}
          >
            <span>⌘</span> Workflows
          </button>
          <button
            className={`nav-item ${activeView === 'models' ? 'active' : ''}`}
            onClick={() => setActiveView('models')}
          >
            <span>▧</span> Models
          </button>
        </div>

        <div className="sidebar-section history-section">
          <div className="section-label">CONVERSATION HISTORY</div>

          {historyLoading && conversations.length === 0 ? (
            <div className="empty-history">Loading history...</div>
          ) : conversations.length > 0 ? (
            <div className="conversation-list">
              {conversations.map((conversation) => (
                <div
                  className={`conversation-row ${
                    activeConversation === conversation.id ? 'selected' : ''
                  }`}
                  key={conversation.id}
                >
                  <button
                    className="conversation-select"
                    onClick={() => void openConversation(conversation.id)}
                    disabled={loading}
                    title={conversation.title}
                  >
                    <span className="conversation-title">{conversation.title}</span>
                    <span className="conversation-count">
                      {conversation.message_count} messages
                    </span>
                  </button>
                  <button
                    className="conversation-delete"
                    onClick={() => void deleteConversation(conversation.id)}
                    disabled={loading}
                    title="Delete conversation"
                    aria-label={`Delete ${conversation.title}`}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-history">No conversations yet</div>
          )}

          {historyError && <div className="history-error">{historyError}</div>}
        </div>

        <div className="sidebar-bottom">
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
            <strong>{activeView === 'documents' ? 'Documents' : activeView === 'workflows' ? 'Workflows' : activeView === 'models' ? 'Models' : 'Chat'}</strong>
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

        {activeView === 'documents' ? (
          <section className="documents-area">
            <div className="documents-heading">
              <div>
                <div className="eyebrow">KNOWLEDGE BASE</div>
                <h1>Documents</h1>
                <p>Manage the PDFs SALAZAR can retrieve information from.</p>
              </div>
              <button
                className="upload-button"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading || !backendOnline}
              >
                {uploading ? 'Indexing...' : '+ Add PDF'}
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,application/pdf"
                onChange={(event) => void uploadDocument(event)}
                hidden
              />
            </div>

            <div className="document-notice">
              <span>◈</span>
              PDFs are processed locally. Text is extracted, chunked, embedded,
              and added to SALAZAR's searchable index.
            </div>

            {documentMessage && (
              <div className="document-success">{documentMessage}</div>
            )}
            {documentError && (
              <div className="document-error">{documentError}</div>
            )}

            <div className="documents-toolbar">
              <div>
                <strong>Your library</strong>
                <span>{documents.length} document{documents.length === 1 ? '' : 's'}</span>
              </div>
              <button
                className="refresh-documents"
                onClick={() => void refreshDocuments()}
                disabled={documentsLoading || uploading}
              >
                {documentsLoading ? 'Refreshing...' : '↻ Refresh'}
              </button>
            </div>

            {documentsLoading && documents.length === 0 ? (
              <div className="documents-empty">Loading your documents...</div>
            ) : documents.length === 0 ? (
              <div className="documents-empty">
                <div className="documents-empty-icon">▤</div>
                <strong>No documents yet</strong>
                <p>Add a PDF to build your private knowledge base.</p>
                <button
                  className="upload-button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={uploading || !backendOnline}
                >
                  + Upload your first PDF
                </button>
              </div>
            ) : (
              <div className="document-list">
                {documents.map((document) => (
                  <article className="document-card" key={document.filename}>
                    <div className="document-icon">PDF</div>
                    <div className="document-details">
                      <strong title={document.filename}>{document.filename}</strong>
                      <span>
                        {(document.size_bytes / (1024 * 1024)).toFixed(2)} MB
                        <span className="document-separator">·</span>
                        {document.chunks} chunks
                      </span>
                    </div>
                    <span className={`document-status ${document.indexed ? 'indexed' : 'pending'}`}>
                      {document.indexed ? 'Indexed' : 'Not indexed'}
                    </span>
                    <button
                      className="document-delete"
                      onClick={() => void removeDocument(document.filename)}
                      disabled={uploading}
                      title={`Delete ${document.filename}`}
                      aria-label={`Delete ${document.filename}`}
                    >
                      ×
                    </button>
                  </article>
                ))}
              </div>
            )}
            <div className="documents-footnote">
              Maximum PDF size: 50 MB. Scanned/image-only PDFs may not contain extractable text.
            </div>
          </section>
        ) : activeView === 'workflows' ? (
          <section className="documents-area">
            <div className="documents-heading">
              <div>
                <div className="eyebrow">AUTOMATION</div>
                <h1>Workflows</h1>
                <p>Multi-agent workflows powered by SALAZAR's existing orchestration engine.</p>
              </div>
            </div>

            <div className="document-notice">
              <span>◈</span>
              Choose a workflow to prepare its request in Chat. Execution uses the existing local Chat API.
            </div>

            <div className="workflow-panel">
              <article className="workflow-card">
                <div className="workflow-card-top">
                  <span className="document-icon">RAG</span>
                  <span className="document-status indexed">Available</span>
                </div>
                <h2>Document Research &amp; Synthesis</h2>
                <p>Retrieves relevant passages from indexed PDFs and synthesizes a source-grounded answer.</p>
                <div className="workflow-steps">
                  <span>01 · Document RAG Agent</span>
                  <span>02 · Local Chat Agent</span>
                </div>
                <label className="workflow-label" htmlFor="workflow-topic">Research topic or question</label>
                <input
                  id="workflow-topic"
                  className="workflow-input"
                  value={workflowTopic}
                  onChange={(event) => setWorkflowTopic(event.target.value)}
                  placeholder="e.g. Explain the system architecture"
                />
                <button
                  className="upload-button"
                  disabled={!workflowTopic.trim() || !backendOnline || loading}
                  onClick={() => {
                    setInput(`Research my documents for ${workflowTopic.trim()} and summarize the findings.`);
                    setActiveView('chat');
                  }}
                >
                  Prepare in Chat →
                </button>
              </article>

              <article className="workflow-card">
                <div className="workflow-card-top">
                  <span className="document-icon">CODE</span>
                  <span className="document-status indexed">Available</span>
                </div>
                <h2>Code Review &amp; Explanation</h2>
                <p>Passes supplied code to the Coding Agent, then asks the Local Chat Agent to explain the findings.</p>
                <div className="workflow-steps">
                  <span>01 · Coding Agent</span>
                  <span>02 · Local Chat Agent</span>
                </div>
                <label className="workflow-label" htmlFor="workflow-code">Paste code to review</label>
                <textarea
                  id="workflow-code"
                  className="workflow-input workflow-code-input"
                  value={workflowCode}
                  onChange={(event) => setWorkflowCode(event.target.value)}
                  placeholder="Paste your code here..."
                />
                <button
                  className="upload-button"
                  disabled={!workflowCode.trim() || !backendOnline || loading}
                  onClick={() => {
                    setInput([
                      'Review my code and explain the findings and proposed fixes.',
                      '',
                      '```',
                      workflowCode.trim(),
                      '```'
                    ].join('\n'));
                    setActiveView('chat');
                  }}
                >
                  Prepare in Chat →
                </button>
              </article>
            </div>
            <div className="documents-footnote">
              Selecting a workflow prepares the request. Press Send in Chat to execute it.
            </div>
          </section>
        ) : activeView === 'models' ? (
          <section className="documents-area models-area">
            <div className="documents-heading models-heading">
              <div>
                <div className="eyebrow">LOCAL MODEL REGISTRY</div>
                <h1>Models</h1>
                <p>Local AI assets bundled with this SALAZAR workstation.</p>
              </div>
              <div className="models-count">
                <strong>03</strong>
                <span>ASSETS BUNDLED</span>
              </div>
            </div>

            <div className="models-notice">
              <span>◈</span>
              <p>These assets are stored locally. Model switching and live load-state reporting are not available in this build.</p>
            </div>

            <div className="model-grid">
              <article className="model-card">
                <div className="model-card-top">
                  <span className="model-type">LANGUAGE MODEL</span>
                  <span className="model-status">● Installed</span>
                </div>
                <div className="model-card-title">
                  <span className="model-icon">LLM</span>
                  <div>
                    <h2>SmolLM2 1.7B Instruct</h2>
                    <p>Primary local language model for conversational inference.</p>
                  </div>
                </div>
                <div className="model-meta">
                  <div><span>Format</span><strong>GGUF</strong></div>
                  <div><span>Quantization</span><strong>Q4_K_M</strong></div>
                  <div className="model-path"><span>Storage path</span><code>models/llm/smollm2-1.7b/</code></div>
                </div>
              </article>

              <article className="model-card">
                <div className="model-card-top">
                  <span className="model-type">EMBEDDING MODEL</span>
                  <span className="model-status">● Installed</span>
                </div>
                <div className="model-card-title">
                  <span className="model-icon">EMB</span>
                  <div>
                    <h2>BGE Small EN v1.5</h2>
                    <p>Represents document passages as vectors for semantic retrieval.</p>
                  </div>
                </div>
                <div className="model-meta">
                  <div><span>Type</span><strong>Embedding model</strong></div>
                  <div className="model-path"><span>Storage path</span><code>models/embeddings/bge-small-en-v1.5/</code></div>
                </div>
              </article>

              <article className="model-card">
                <div className="model-card-top">
                  <span className="model-type">RERANKER</span>
                  <span className="model-status model-status-muted">● Assets present</span>
                </div>
                <div className="model-card-title">
                  <span className="model-icon">RER</span>
                  <div>
                    <h2>MiniLM Reranker</h2>
                    <p>Reranker model assets are bundled with the workstation.</p>
                  </div>
                </div>
                <div className="model-meta">
                  <div><span>Type</span><strong>Reranker</strong></div>
                  <div className="model-path"><span>Storage path</span><code>models/reranker/minilm/</code></div>
                </div>
                <div className="model-runtime-note">Runtime activation is not reported by the current API.</div>
              </article>
            </div>

            <div className="models-footnote">
              “Installed” confirms that model files are present on the workstation; it does not indicate a live model-load state.
            </div>
          </section>
        ) : (
          <>
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
                <article className={`message ${message.role}`} key={message.id}>
                  <div className="message-avatar">
                    {message.role === 'user' ? 'A' : 'S'}
                  </div>
                  <div className="message-body">
                    <div className="message-heading">
                      <strong>{message.role === 'user' ? 'You' : 'SALAZAR'}</strong>
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
          </>
        )}
      </main>
    </div>
  );
}

export default App;
