import React, { useState, useEffect, useRef } from 'react';

/**
 * SmartChatGPT - A React-only chatbot component with:
 * - Animated typing for the assistant
 * - Simple intent detection and short-lived context
 * - No backend or external API calls
 *
 * Notes:
 * - Uses inline styles and a small <style> block for typing dots animation.
 * - Keeps to a clean, modern style that fits in with an "Ocean Professional" style palette.
 * - No environment variables required.
 */

// Intent keywords for simple rule-based responses
const INTENTS = [
  { key: 'greeting', patterns: ['hello', 'hi', 'hey'], response: 'Hi there! 👋 How can I help you today?' },
  { key: 'help', patterns: ['help', 'assist', 'support'], response: 'Sure, tell me what you need help with and I can guide you.' },
  { key: 'thanks', patterns: ['thank', 'thanks', 'thx'], response: 'You’re very welcome! 😊 Anything else I can do?' },
  { key: 'bye', patterns: ['bye', 'goodbye', 'see you'], response: 'Goodbye! 👋 Have a great day!' },
  { key: 'theme', patterns: ['theme', 'color', 'design'], response: 'We’re using a modern ocean-inspired style with blue and amber accents for clarity and focus.' },
];

// Simple context tracker (short-lived) to slightly vary responses
function updateContext(context, userText) {
  const lower = userText.toLowerCase();
  const next = { ...context };
  if (lower.includes('help')) next.lastIntent = 'help';
  else if (lower.includes('hello') || lower.includes('hi') || lower.includes('hey')) next.lastIntent = 'greeting';
  else if (lower.includes('thanks')) next.lastIntent = 'thanks';
  else if (lower.includes('bye')) next.lastIntent = 'bye';
  else if (lower.includes('theme') || lower.includes('color')) next.lastIntent = 'theme';
  else next.lastIntent = 'general';
  // Very short-lived: keep only last 3 user messages
  const history = (next.history || []).slice(-2);
  next.history = [...history, userText];
  return next;
}

function decideResponse(userText, context) {
  const lower = userText.toLowerCase();
  // Try intent matches
  for (const intent of INTENTS) {
    if (intent.patterns.some(p => lower.includes(p))) {
      // Tiny contextual variation
      if (context.lastIntent === intent.key) {
        return `${intent.response} By the way, I recall we were on the same topic earlier.`;
      }
      return intent.response;
    }
  }
  // Fallbacks based on minimal context
  if (context.lastIntent === 'help') {
    return 'Could you share more specifics? For example: the goal, constraints, or where you’re stuck.';
  }
  if (lower.endsWith('?')) {
    return 'Great question! I can help you explore possible approaches or explain the concepts involved.';
  }
  return "Got it. Would you like a summary, suggestions, or examples to move forward?";
}

// PUBLIC_INTERFACE
export default function SmartChatGPT() {
  /** PUBLIC_INTERFACE
   * Main SmartChatGPT component rendering the chat UI with typing animation and basic logic.
   */
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I’m SmartChatGPT 🤖 — a lightweight, React-only demo. How can I help you?' },
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [context, setContext] = useState({ lastIntent: null, history: [] });
  const listRef = useRef(null);
  const inputRef = useRef(null);

  // Keep scroll pinned to bottom
  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || isTyping) return;

    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setInput('');
    setIsTyping(true);

    // Simulate "thinking" and animated typing
    const updatedContext = updateContext(context, text);
    setContext(updatedContext);

    // Delay to simulate processing
    await new Promise(r => setTimeout(r, 400));
    const reply = decideResponse(text, updatedContext);

    // Simulate typing effect
    await typeOutAssistantMessage(reply);
    setIsTyping(false);
    inputRef.current?.focus();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Simulates animated typing by progressively appending characters
  const typeOutAssistantMessage = async (fullText) => {
    const typingId = Math.random().toString(36);
    setMessages(prev => [...prev, { role: 'assistant', content: '', typingId }]);

    let current = '';
    for (let i = 0; i < fullText.length; i++) {
      current += fullText[i];
      // Randomized typing cadence for more natural feel
      const delay = 12 + Math.random() * 25;
      // eslint-disable-next-line no-await-in-loop
      await new Promise(r => setTimeout(r, delay));
      setMessages(prev => prev.map(m => {
        if (m.typingId === typingId) return { ...m, content: current };
        return m;
      }));
    }

    // Remove typingId flag at the end
    setMessages(prev => prev.map(m => {
      if (m.typingId === typingId) {
        const { typingId: _omit, ...rest } = m;
        return rest;
      }
      return m;
    }));
  };

  // Basic styles aligned with "Ocean Professional" and existing app theme approach
  const styles = {
    page: {
      background: 'var(--bg-primary, #f9fafb)',
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: 16,
      color: 'var(--text-primary, #111827)',
    },
    card: {
      width: '100%',
      maxWidth: 880,
      background: 'var(--bg-secondary, #ffffff)',
      border: '1px solid var(--border-color, #E5E7EB)',
      borderRadius: 16,
      boxShadow: '0 10px 30px rgba(17, 24, 39, 0.08)',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    },
    header: {
      padding: '18px 20px',
      borderBottom: '1px solid var(--border-color, #E5E7EB)',
      background: 'linear-gradient(135deg, rgba(37,99,235,0.06) 0%, rgba(249,250,251,1) 100%)',
    },
    title: {
      display: 'flex',
      alignItems: 'center',
      gap: 10,
      margin: 0,
      fontSize: 20,
      fontWeight: 700,
      color: '#2563EB',
    },
    badge: {
      fontSize: 11,
      fontWeight: 700,
      background: '#F59E0B',
      color: '#111827',
      padding: '2px 8px',
      borderRadius: 999,
      letterSpacing: 0.2,
    },
    subtitle: {
      margin: '6px 0 0',
      fontSize: 13,
      color: 'var(--text-secondary, #4B5563)',
    },
    body: {
      display: 'flex',
      flexDirection: 'column',
      height: '60vh',
      background: 'var(--bg-primary, #f9fafb)',
    },
    messages: {
      flex: 1,
      overflowY: 'auto',
      padding: 16,
    },
    row: {
      display: 'flex',
      marginBottom: 12,
    },
    bubbleAssistant: {
      marginRight: 'auto',
      background: '#ffffff',
      color: 'var(--text-primary, #111827)',
      border: '1px solid var(--border-color, #E5E7EB)',
      padding: '10px 14px',
      borderRadius: 14,
      borderTopLeftRadius: 4,
      maxWidth: '75%',
      lineHeight: 1.45,
      boxShadow: '0 6px 16px rgba(17,24,39,0.06)',
      wordBreak: 'break-word',
      whiteSpace: 'pre-wrap',
    },
    bubbleUser: {
      marginLeft: 'auto',
      background: '#2563EB',
      color: '#ffffff',
      padding: '10px 14px',
      borderRadius: 14,
      borderTopRightRadius: 4,
      maxWidth: '75%',
      lineHeight: 1.45,
      boxShadow: '0 6px 16px rgba(37,99,235,0.2)',
      wordBreak: 'break-word',
      whiteSpace: 'pre-wrap',
    },
    footer: {
      display: 'flex',
      alignItems: 'center',
      gap: 10,
      borderTop: '1px solid var(--border-color, #E5E7EB)',
      background: 'var(--bg-secondary, #ffffff)',
      padding: 12,
    },
    input: {
      flex: 1,
      fontSize: 15,
      padding: '12px 14px',
      borderRadius: 12,
      border: '1px solid var(--border-color, #E5E7EB)',
      outline: 'none',
      background: '#ffffff',
      color: 'var(--text-primary, #111827)',
      boxShadow: '0 1px 2px rgba(17,24,39,0.04) inset',
    },
    button: {
      fontSize: 15,
      fontWeight: 600,
      padding: '12px 18px',
      borderRadius: 12,
      border: 'none',
      background: '#2563EB',
      color: '#ffffff',
      cursor: 'pointer',
      boxShadow: '0 8px 20px rgba(37,99,235,0.25)',
      transition: 'transform 0.08s ease, box-shadow 0.2s ease, opacity 0.2s ease',
    },
    buttonDisabled: {
      background: '#93C5FD',
      boxShadow: 'none',
      cursor: 'not-allowed',
      opacity: 0.8,
    },
    typing: {
      display: 'inline-flex',
      alignItems: 'center',
      gap: 4,
      color: 'var(--text-secondary, #6B7280)',
      fontSize: 13,
      marginLeft: 2,
    },
  };

  return (
    <div style={styles.page}>
      {/* Local style for animated typing dots */}
      <style>{`
        @keyframes blink {
          0% { opacity: 0.2; }
          20% { opacity: 1; }
          100% { opacity: 0.2; }
        }
        .dot {
          width: 6px;
          height: 6px;
          background: #6B7280;
          border-radius: 50%;
          display: inline-block;
          animation: blink 1.2s infinite;
        }
        .dot:nth-child(2) { animation-delay: 0.2s; }
        .dot:nth-child(3) { animation-delay: 0.4s; }
      `}</style>

      <section style={styles.card} aria-label="SmartChatGPT chat window">
        <header style={styles.header}>
          <h1 style={styles.title}>
            SmartChatGPT
            <span style={styles.badge}>React-only</span>
          </h1>
          <p style={styles.subtitle}>Lightweight assistant with animated typing and simple intents</p>
        </header>

        <div style={styles.body}>
          <div ref={listRef} style={styles.messages} aria-live="polite">
            {messages.map((m, idx) => (
              <div key={idx} style={styles.row}>
                <div style={m.role === 'user' ? styles.bubbleUser : styles.bubbleAssistant}>
                  {m.content}
                  {m.role === 'assistant' && idx === messages.length - 1 && isTyping ? (
                    <span style={styles.typing} aria-label="Assistant is typing">
                      <span className="dot" />
                      <span className="dot" />
                      <span className="dot" />
                    </span>
                  ) : null}
                </div>
              </div>
            ))}
            {isTyping && messages[messages.length - 1]?.role !== 'assistant' ? (
              <div style={styles.row}>
                <div style={styles.bubbleAssistant}>
                  <span style={styles.typing} aria-label="Assistant is typing">
                    <span className="dot" />
                    <span className="dot" />
                    <span className="dot" />
                  </span>
                </div>
              </div>
            ) : null}
          </div>

          <footer style={styles.footer}>
            <input
              ref={inputRef}
              type="text"
              placeholder="Type a message and press Enter…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              style={styles.input}
              aria-label="Message input"
            />
            <button
              type="button"
              onClick={handleSend}
              disabled={isTyping || !input.trim()}
              style={{
                ...styles.button,
                ...(isTyping || !input.trim() ? styles.buttonDisabled : {}),
              }}
              aria-label="Send message"
            >
              {isTyping ? 'Thinking…' : 'Send'}
            </button>
          </footer>
        </div>
      </section>
    </div>
  );
}
