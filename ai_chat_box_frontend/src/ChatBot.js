import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

/**
 * SmartTalk ChatBot implemented purely in React using TensorFlow.js.
 * - No backend calls or external APIs.
 * - Uses a tiny placeholder TF.js model pipeline to "process" messages on-device.
 * - Styled with the Ocean Professional theme palette.
 *
 * Note: This is a demonstrative, client-only chatbot. The "ML" here is a lightweight
 * placeholder using TensorFlow.js tensors to simulate tokenization/embedding and a
 * simple rule-based response. Real models would be much heavier and likely require
 * a dedicated model file or WebGL/WASM backends.
 */

// Ocean Professional palette
const COLORS = {
  primary: '#2563EB',   // blue
  secondary: '#F59E0B', // amber
  bg: '#f9fafb',
  surface: '#ffffff',
  text: '#111827',
  subtleText: '#4B5563',
  border: '#E5E7EB',
};

// PUBLIC_INTERFACE
export default function ChatBot() {
  /**
   * PUBLIC_INTERFACE
   * SmartTalk component renders a simple chat UI and runs a minimal TFJS-based processor.
   */
  const [messages, setMessages] = useState([
    { from: 'bot', text: 'Hello! I am SmartTalk 🤖. How can I help you today?' },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const inputRef = useRef(null);
  const listRef = useRef(null);

  // "Initialize" a trivial model-like object once.
  const model = useMemo(() => {
    // This is a fake demo "model": maps characters to numbers and computes a soft score.
    // In a real use-case, load a tfjs model via tf.loadLayersModel or tf.loadGraphModel.
    const charIndex = new Map();
    const alphabet = 'abcdefghijklmnopqrstuvwxyz ';
    [...alphabet].forEach((ch, i) => charIndex.set(ch, i + 1));

    return {
      // Naive "tokenizer"
      tokenize: (text) => {
        const lower = (text || '').toLowerCase();
        const tokens = Array.from(lower).map((ch) => charIndex.get(ch) || 0);
        return tokens.length ? tokens : [0];
      },
      // Naive "inference": create a tensor, do a simple reduction to simulate "understanding"
      infer: (tokens) => {
        const t = tf.tensor1d(tokens, 'float32');
        const sum = t.sum(); // simplistic
        const mean = t.mean();
        const score = tf.add(sum, mean).dataSync()[0]; // number
        t.dispose();
        mean.dispose?.(); // mean is a Tensor
        return score;
      },
    };
  }, []);

  // Keep scroll at bottom when messages change
  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages]);

  // Simulated "thinking" with TensorFlow.js
  const generateBotResponse = async (userText) => {
    // Simple safety check
    const text = (userText || '').trim();
    if (!text) return "I'm here whenever you're ready.";

    // Lightweight async to avoid blocking UI
    await tf.nextFrame();

    const tokens = model.tokenize(text);
    const score = model.infer(tokens);

    // Very naive rule-based responses guided by score and keywords
    const lower = text.toLowerCase();
    if (lower.includes('hello') || lower.includes('hi')) {
      return 'Hi there! 👋 What would you like to chat about?';
    }
    if (lower.includes('help')) {
      return 'Sure! Tell me a bit more about what you need help with.';
    }
    if (lower.includes('color') || lower.includes('theme')) {
      return 'We are using the Ocean Professional theme: blue (#2563EB), amber (#F59E0B), and a clean surface.';
    }

    if (score < 50) {
      return 'Interesting! Could you elaborate a bit more?';
    } else if (score < 150) {
      return 'Got it. That makes sense. Do you have any follow-up questions?';
    }
    return 'Thanks for the details! If you want, I can summarize or suggest next steps.';
  };

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || loading) return;

    setMessages((prev) => [...prev, { from: 'user', text: trimmed }]);
    setInput('');
    setLoading(true);
    try {
      const reply = await generateBotResponse(trimmed);
      setMessages((prev) => [...prev, { from: 'bot', text: reply }]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { from: 'bot', text: 'Oops! I ran into an issue processing that. Please try again.' },
      ]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Styles (Ocean Professional)
  const styles = {
    page: {
      background: COLORS.bg,
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '24px',
      color: COLORS.text,
    },
    card: {
      background: COLORS.surface,
      width: '100%',
      maxWidth: '860px',
      borderRadius: '16px',
      boxShadow: '0 10px 30px rgba(17, 24, 39, 0.08)',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
      border: `1px solid ${COLORS.border}`,
    },
    header: {
      padding: '20px 24px',
      borderBottom: `1px solid ${COLORS.border}`,
      background:
        'linear-gradient(135deg, rgba(37,99,235,0.06) 0%, rgba(249,250,251,1) 100%)',
    },
    title: {
      margin: 0,
      fontSize: '20px',
      fontWeight: 700,
      color: COLORS.primary,
      letterSpacing: '0.2px',
    },
    subtitle: {
      margin: '6px 0 0 0',
      fontSize: '13px',
      color: COLORS.subtleText,
    },
    content: {
      display: 'flex',
      flexDirection: 'column',
      height: '60vh',
    },
    messages: {
      flex: 1,
      overflowY: 'auto',
      padding: '16px',
      background: COLORS.bg,
    },
    row: {
      display: 'flex',
      marginBottom: '12px',
    },
    bubbleUser: {
      marginLeft: 'auto',
      background: COLORS.primary,
      color: '#ffffff',
      padding: '10px 14px',
      borderRadius: '14px',
      borderTopRightRadius: '4px',
      maxWidth: '75%',
      lineHeight: 1.45,
      boxShadow: '0 6px 16px rgba(37,99,235,0.2)',
      wordBreak: 'break-word',
    },
    bubbleBot: {
      marginRight: 'auto',
      background: '#ffffff',
      color: COLORS.text,
      border: `1px solid ${COLORS.border}`,
      padding: '10px 14px',
      borderRadius: '14px',
      borderTopLeftRadius: '4px',
      maxWidth: '75%',
      lineHeight: 1.45,
      boxShadow: '0 6px 16px rgba(17,24,39,0.06)',
      wordBreak: 'break-word',
    },
    footer: {
      borderTop: `1px solid ${COLORS.border}`,
      padding: '12px',
      background: COLORS.surface,
      display: 'flex',
      gap: '10px',
      alignItems: 'center',
    },
    input: {
      flex: 1,
      fontSize: '15px',
      padding: '12px 14px',
      borderRadius: '12px',
      border: `1px solid ${COLORS.border}`,
      outline: 'none',
      background: '#ffffff',
      color: COLORS.text,
      boxShadow: '0 1px 2px rgba(17,24,39,0.04) inset',
    },
    button: {
      fontSize: '15px',
      fontWeight: 600,
      padding: '12px 18px',
      borderRadius: '12px',
      border: 'none',
      background: COLORS.primary,
      color: '#ffffff',
      cursor: 'pointer',
      boxShadow: '0 8px 20px rgba(37,99,235,0.25)',
      transition: 'transform 0.08s ease, box-shadow 0.2s ease',
    },
    buttonDisabled: {
      background: '#93C5FD',
      boxShadow: 'none',
      cursor: 'not-allowed',
    },
    hint: {
      fontSize: '12px',
      color: COLORS.subtleText,
      marginLeft: '4px',
    },
    badge: {
      marginLeft: '8px',
      background: COLORS.secondary,
      color: '#111827',
      fontSize: '11px',
      fontWeight: 700,
      padding: '3px 8px',
      borderRadius: '9999px',
      letterSpacing: '0.3px',
    },
  };

  return (
    <div style={styles.page}>
      <section style={styles.card} aria-label="SmartTalk chat window">
        <header style={styles.header}>
          <h1 style={styles.title}>
            SmartTalk
            <span style={styles.badge}>Beta</span>
          </h1>
          <p style={styles.subtitle}>
            Ocean Professional theme • On-device demo with TensorFlow.js
          </p>
        </header>

        <div style={styles.content}>
          <div ref={listRef} style={styles.messages} aria-live="polite">
            {messages.map((m, idx) => (
              <div key={idx} style={styles.row}>
                <div style={m.from === 'user' ? styles.bubbleUser : styles.bubbleBot}>
                  {m.text}
                </div>
              </div>
            ))}
          </div>

          <footer style={styles.footer}>
            <input
              ref={inputRef}
              type="text"
              placeholder="Type your message and press Enter…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              style={styles.input}
              aria-label="Message input"
            />
            <button
              type="button"
              onClick={handleSend}
              disabled={loading || !input.trim()}
              style={{
                ...styles.button,
                ...(loading || !input.trim() ? styles.buttonDisabled : {}),
              }}
              aria-label="Send message"
            >
              {loading ? 'Thinking…' : 'Send'}
            </button>
            <span style={styles.hint}>Powered by tfjs</span>
          </footer>
        </div>
      </section>
    </div>
  );
}
