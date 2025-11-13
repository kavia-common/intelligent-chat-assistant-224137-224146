import React, { useState, useEffect, useRef } from 'react';

/**
 * SmartChatGPT - A React-only chatbot component enhanced with live in-browser learning.
 * - Animated typing for the assistant (unchanged)
 * - Lightweight intent classifier with online-updatable weights (perceptron-like)
 * - Feature extraction over user text (keywords, cues, length buckets)
 * - Online learning step after each exchange (reinforce chosen intent)
 * - Persistence of weights and stats to localStorage
 * - UI controls to toggle learning and reset memory
 *
 * No backend or external services; all computation in-browser.
 */

// Intent space kept small and rule-friendly; includes baseline rules and trainable bias
const INTENT_DEFS = [
  { key: 'greeting', baseBias: 0.3, response: 'Hi there! 👋 How can I help you today?' },
  { key: 'smalltalk_status', baseBias: 0.1, response: 'I’m doing great! How about you?' },
  { key: 'weather', baseBias: 0.0, response: 'I don’t pull live weather, but it looks like a great day to build something cool! 🌤️' },
  { key: 'name', baseBias: 0.05, response: 'I’m SmartChatGPT — a lightweight assistant running entirely in your browser.' },
  { key: 'farewell', baseBias: 0.2, response: 'Goodbye! 👋 Have a great day!' },
  { key: 'project_context', baseBias: 0.1, response: 'We’re using an Ocean Professional theme with a modern React UI. What would you like to build?' },
  { key: 'generic_followup', baseBias: 0.15, response: 'Got it. Would you like a summary, suggestions, or examples to move forward?' },
];

// Token groups and cues for feature extraction
const FEATURE_SPECS = [
  // keyword groups
  { name: 'kw_hello', tokens: ['hello', 'hi', 'hey', 'yo'], weight: 1 },
  { name: 'kw_how_are_you', tokens: ['how are you', 'hows it going', 'how r u'], weight: 1 },
  { name: 'kw_weather', tokens: ['weather', 'rain', 'sunny', 'forecast'], weight: 1 },
  { name: 'kw_name', tokens: ['name', 'who are you', 'what are you'], weight: 1 },
  { name: 'kw_thanks', tokens: ['thanks', 'thank you', 'thx', 'appreciate'], weight: 1 },
  { name: 'kw_bye', tokens: ['bye', 'goodbye', 'see you', 'cya'], weight: 1 },
  { name: 'kw_project', tokens: ['project', 'build', 'tech', 'react', 'frontend', 'design', 'theme'], weight: 1 },
  { name: 'kw_help', tokens: ['help', 'assist', 'support'], weight: 1 },

  // punctuation cues
  { name: 'cue_question', tokens: ['?'], weight: 1 },
  { name: 'cue_exclaim', tokens: ['!'], weight: 1 },

  // sentiment-ish cues
  { name: 'sent_pos', tokens: [':)', '🙂', '😊', '👍', 'great', 'awesome'], weight: 1 },
  { name: 'sent_neg', tokens: [':(', '🙁', '😔', 'bad', 'terrible'], weight: 1 },

  // message length buckets (computed in extractor)
  { name: 'len_short', tokens: [], weight: 1 },  // <= 3 words
  { name: 'len_medium', tokens: [], weight: 1 }, // 4-12 words
  { name: 'len_long', tokens: [], weight: 1 },   // > 12 words
];

// Map from features to baseline bias per intent (rule-ish nudges)
const BASE_RULES = {
  greeting: ['kw_hello', 'sent_pos', 'len_short'],
  smalltalk_status: ['kw_how_are_you', 'sent_pos'],
  weather: ['kw_weather', 'cue_question'],
  name: ['kw_name', 'cue_question'],
  farewell: ['kw_bye', 'sent_pos', 'len_short'],
  project_context: ['kw_project', 'kw_help'],
  generic_followup: ['cue_question', 'len_long', 'kw_help'],
};

// Storage keys
const LS_KEYS = {
  weights: 'smartgpt_intent_weights_v1',
  stats: 'smartgpt_stats_v1',
  learnEnabled: 'smartgpt_learning_enabled_v1',
};

// Utility: clamp a number
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

/**
 * Create zero-initialized weight matrix: { intentKey: number[featureCount] }
 */
function createZeroWeights() {
  const size = FEATURE_SPECS.length;
  const w = {};
  for (const intent of INTENT_DEFS) {
    w[intent.key] = Array(size).fill(0);
  }
  return w;
}

/**
 * Load weights and stats from localStorage; returns safe defaults if absent or invalid.
 */
function loadMemory() {
  try {
    const weightsJson = localStorage.getItem(LS_KEYS.weights);
    const statsJson = localStorage.getItem(LS_KEYS.stats);
    const learnJson = localStorage.getItem(LS_KEYS.learnEnabled);

    const weights = weightsJson ? JSON.parse(weightsJson) : createZeroWeights();
    const stats = statsJson ? JSON.parse(statsJson) : { messages: 0, updates: 0 };
    const learningEnabled = learnJson ? JSON.parse(learnJson) : true;

    // Validate shape
    const featureCount = FEATURE_SPECS.length;
    for (const intent of INTENT_DEFS) {
      if (!Array.isArray(weights[intent.key]) || weights[intent.key].length !== featureCount) {
        weights[intent.key] = Array(featureCount).fill(0);
      }
    }
    return { weights, stats, learningEnabled };
  } catch {
    return { weights: createZeroWeights(), stats: { messages: 0, updates: 0 }, learningEnabled: true };
  }
}

/**
 * Persist weights and stats to localStorage.
 */
function persistMemory(weights, stats, learningEnabled) {
  try {
    localStorage.setItem(LS_KEYS.weights, JSON.stringify(weights));
    localStorage.setItem(LS_KEYS.stats, JSON.stringify(stats));
    localStorage.setItem(LS_KEYS.learnEnabled, JSON.stringify(learningEnabled));
  } catch {
    // ignore quota or serialization errors for this demo
  }
}

/**
 * Extract a dense binary feature vector for the input text.
 * Returns Float32Array length = FEATURE_SPECS.length
 */
function extractFeatures(textRaw) {
  const text = (textRaw || '').toLowerCase().trim();
  const words = text.split(/\s+/).filter(Boolean);
  const vec = new Float32Array(FEATURE_SPECS.length);

  FEATURE_SPECS.forEach((spec, idx) => {
    if (spec.name.startsWith('len_')) {
      // handle later after base token features
      return;
    }
    const hit = spec.tokens.some(tok => text.includes(tok));
    vec[idx] = hit ? 1 : 0;
  });

  // length buckets
  const lenShortIdx = FEATURE_SPECS.findIndex(f => f.name === 'len_short');
  const lenMediumIdx = FEATURE_SPECS.findIndex(f => f.name === 'len_medium');
  const lenLongIdx = FEATURE_SPECS.findIndex(f => f.name === 'len_long');
  const wc = words.length;
  if (wc <= 3 && lenShortIdx >= 0) vec[lenShortIdx] = 1;
  else if (wc <= 12 && lenMediumIdx >= 0) vec[lenMediumIdx] = 1;
  else if (lenLongIdx >= 0) vec[lenLongIdx] = 1;

  return vec;
}

/**
 * Compute score for each intent: dot(W_i, x) + baseBias + ruleBias(features)
 * Returns array of { key, score }
 */
function scoreIntents(weights, features) {
  const scores = [];
  for (const intent of INTENT_DEFS) {
    const w = weights[intent.key] || [];
    let dot = 0;
    for (let i = 0; i < features.length; i++) {
      dot += (w[i] || 0) * features[i];
    }

    // rule bias: add small bumps for rule-aligned features
    const ruleFeats = BASE_RULES[intent.key] || [];
    let ruleBonus = 0;
    for (const fname of ruleFeats) {
      const idx = FEATURE_SPECS.findIndex(f => f.name === fname);
      if (idx >= 0 && features[idx] > 0) {
        ruleBonus += 0.15; // small nudge
      }
    }

    const score = dot + (intent.baseBias || 0) + ruleBonus;
    scores.push({ key: intent.key, score });
  }
  return scores.sort((a, b) => b.score - a.score);
}

/**
 * Online update: perceptron-like reinforcement for chosen intent.
 * weights[intentChosen] += lr * features
 * Optional mild decay for others.
 */
function updateWeights(weights, intentKey, features, lr = 0.1, clampRange = 2.0, decay = 0.01) {
  const featureCount = features.length;
  const copy = { ...weights };

  // reinforce chosen
  const wChosen = (copy[intentKey] || Array(featureCount).fill(0)).slice();
  for (let i = 0; i < featureCount; i++) {
    const val = wChosen[i] + lr * features[i];
    wChosen[i] = clamp(val, -clampRange, clampRange);
  }
  copy[intentKey] = wChosen;

  // mild decay on others to avoid runaway bias
  for (const intent of INTENT_DEFS) {
    if (intent.key === intentKey) continue;
    const w = (copy[intent.key] || Array(featureCount).fill(0)).slice();
    for (let i = 0; i < featureCount; i++) {
      const v = w[i] * (1 - decay);
      w[i] = Math.abs(v) < 1e-4 ? 0 : v;
    }
    copy[intent.key] = w;
  }
  return copy;
}

/**
 * Generate a response string for the chosen intent, with slight variants from context.
 */
function generateResponseForIntent(intentKey, context) {
  const def = INTENT_DEFS.find(d => d.key === intentKey) || INTENT_DEFS.find(d => d.key === 'generic_followup');

  // Tiny contextual variation
  if (context.lastIntent === intentKey) {
    return `${def.response} By the way, I recall we were on the same topic earlier.`;
  }
  if (intentKey === 'greeting' && context.history?.length > 0) {
    return `${def.response} I remember our recent chat—ready to continue?`;
  }
  if (intentKey === 'generic_followup' && (context.lastIntent === 'project_context' || context.lastIntent === 'help')) {
    return 'Makes sense. Would examples or a quick outline help you proceed?';
  }
  return def.response;
}

// Simple context tracker (short-lived) to slightly vary responses
function updateContext(context, userText, decidedIntent) {
  const next = { ...context };
  next.lastIntent = decidedIntent || context.lastIntent || null;
  const history = (next.history || []).slice(-2);
  next.history = [...history, userText];
  return next;
}

// PUBLIC_INTERFACE
export default function SmartChatGPT() {
  /** PUBLIC_INTERFACE
   * Main SmartChatGPT component rendering the chat UI with typing animation and adaptive learning.
   */
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I’m SmartChatGPT 🤖 — now with simple in-browser learning. How can I help?' },
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [context, setContext] = useState({ lastIntent: null, history: [] });

  // learning state
  const [{ weights, stats, learningEnabled }, setMemory] = useState(() => loadMemory());

  const listRef = useRef(null);
  const inputRef = useRef(null);

  // Keep scroll pinned to bottom
  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  // Persist memory on change
  useEffect(() => {
    persistMemory(weights, stats, learningEnabled);
  }, [weights, stats, learningEnabled]);

  // Determine best intent given features and rule bias
  const decideIntent = (text) => {
    const feats = extractFeatures(text);
    const ranked = scoreIntents(weights, feats);

    // Soft check: If multiple tied, prefer ones that match explicit keyword rules
    const top = ranked[0];
    return { topIntent: top.key, features: feats, ranked };
  };

  // Handle learning toggle
  const toggleLearning = () => {
    setMemory((prev) => ({ ...prev, learningEnabled: !prev.learningEnabled }));
  };

  // Handle reset learning
  const resetLearning = () => {
    const fresh = createZeroWeights();
    const newStats = { messages: 0, updates: 0 };
    setMemory({ weights: fresh, stats: newStats, learningEnabled: true });
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || isTyping) return;

    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setInput('');
    setIsTyping(true);

    // Simulate processing delay
    await new Promise(r => setTimeout(r, 250));

    // Decide intent using features + learned weights + baseline rules
    const { topIntent, features } = decideIntent(text);

    // Update context to include intent
    const updatedContext = updateContext(context, text, topIntent);
    setContext(updatedContext);

    // Generate response biased by chosen intent (keeping existing typing effect)
    const reply = generateResponseForIntent(topIntent, updatedContext);

    // Online update after the exchange (reinforce chosen intent)
    setMemory((prev) => {
      const nextStats = { messages: (prev.stats?.messages || 0) + 1, updates: prev.stats?.updates || 0 };
      if (prev.learningEnabled) {
        const nextWeights = updateWeights(prev.weights, topIntent, features, 0.1, 2.0, 0.01);
        const incStats = { ...nextStats, updates: nextStats.updates + 1 };
        return { ...prev, weights: nextWeights, stats: incStats };
      }
      return { ...prev, stats: nextStats };
    });

    // Typing animation for reply text
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
      const delay = 12 + Math.random() * 25;
      // eslint-disable-next-line no-await-in-loop
      await new Promise(r => setTimeout(r, delay));
      setMessages(prev => prev.map(m => (m.typingId === typingId ? { ...m, content: current } : m)));
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
      flexWrap: 'wrap',
    },
    input: {
      flex: 1,
      minWidth: 180,
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
    buttonSecondary: {
      background: '#F59E0B',
      color: '#111827',
      boxShadow: '0 8px 20px rgba(245,158,11,0.25)',
    },
    buttonDanger: {
      background: '#EF4444',
      color: '#ffffff',
      boxShadow: '0 8px 20px rgba(239,68,68,0.25)',
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
    learnBar: {
      width: '100%',
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      marginTop: 8,
    },
    statText: {
      fontSize: 12,
      color: 'var(--text-secondary, #6B7280)',
      marginLeft: 'auto',
    },
    toggleLabel: {
      fontSize: 13,
      color: 'var(--text-secondary, #374151)',
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
          <p style={styles.subtitle}>
            Lightweight assistant with animated typing and adaptive, in-browser learning
          </p>
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

            {/* Learning controls bar */}
            <div style={styles.learnBar} role="group" aria-label="Learning controls">
              <label style={styles.toggleLabel}>
                <input
                  type="checkbox"
                  checked={learningEnabled}
                  onChange={toggleLearning}
                  aria-label="Toggle online learning"
                  style={{ marginRight: 6 }}
                />
                Learning: {learningEnabled ? 'On' : 'Off'}
              </label>

              <button
                type="button"
                onClick={resetLearning}
                style={{ ...styles.button, ...styles.buttonDanger }}
                aria-label="Reset learning memory"
                title="Clear learned weights and stats"
              >
                Reset Learning
              </button>

              <span style={styles.statText} aria-live="polite">
                Msgs: {stats?.messages ?? 0} • Updates: {stats?.updates ?? 0}
              </span>
            </div>
          </footer>
        </div>
      </section>
    </div>
  );
}
