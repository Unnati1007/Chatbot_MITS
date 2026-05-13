import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, RefreshCw, MessageSquare, Globe, BookOpen, GraduationCap, Mic, MicOff, Star, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const API_URL = 'http://127.0.0.1:5000/chat';

function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! 👋 I'm your **MITS AI Assistant**. Ask me anything about Moodle, IMS, or Registration.",
      sender: 'bot',
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      type: 'rule',
      suggestions: ["How to get Moodle ID?", "Forgot Password", "IMS Registration"]
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [view, setView] = useState(window.location.pathname === '/admin' ? 'admin' : 'chat');
  const [adminData, setAdminData] = useState([]);
  const [adminStats, setAdminStats] = useState([]);
  const [adminTab, setAdminTab] = useState('feedback'); // 'feedback' or 'learning'
  const messagesEndRef = useRef(null);

  const formatDate = (dateStr) => {
    if (!dateStr) return "Recently";
    const date = new Date(dateStr);
    return isNaN(date.getTime()) ? "Recently" : date.toLocaleString();
  };

  // --- ROUTING LOGIC ---
  useEffect(() => {
    const handleLocationChange = () => {
      setView(window.location.pathname === '/admin' ? 'admin' : 'chat');
    };
    window.addEventListener('popstate', handleLocationChange);
    return () => window.removeEventListener('popstate', handleLocationChange);
  }, []);

  const navigate = (path) => {
    window.history.pushState({}, '', path);
    setView(path === '/admin' ? 'admin' : 'chat');
  };

  const fetchAdminData = () => {
    // Fetch feedback
    fetch('http://127.0.0.1:5000/api/admin/feedback')
      .then(res => res.json())
      .then(data => setAdminData(data))
      .catch(err => console.error("Admin fetch failed", err));

    // Fetch RL stats
    fetch('http://127.0.0.1:5000/api/admin/stats')
      .then(res => res.json())
      .then(data => setAdminStats(data))
      .catch(err => console.error("Stats fetch failed", err));
  };

  // --- ADMIN DATA FETCH ---
  useEffect(() => {
    if (view === 'admin') {
      fetchAdminData();
    }
  }, [view]);

  // --- VOICE RECOGNITION LOGIC ---
  const handleVoiceInput = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      alert("Your browser does not support voice recognition. Please try Chrome or Edge.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = 'en-IN'; // Optimized for Indian English
    recognition.interimResults = false;

    recognition.onstart = () => setIsListening(true);
    recognition.onend = () => setIsListening(false);
    
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setInput(transcript);
    };

    if (isListening) {
      recognition.stop();
    } else {
      recognition.start();
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSend = async (messageText = input) => {
    const text = messageText.trim();
    if (!text || isLoading) return;

    const userMsg = {
      id: Date.now(),
      text,
      sender: 'user',
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:5000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text })
      });

      const contentType = response.headers.get('content-type');

      // --- HANDLE JSON (Rules/Greetings) ---
      if (contentType && contentType.includes('application/json')) {
        const data = await response.json();
        setMessages(prev => [...prev, {
          id: Date.now() + 1,
          text: data.reply,
          sender: 'bot',
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          type: data.type
        }]);
        setIsLoading(false);
        return;
      }

      // --- HANDLE STREAMING (AI) ---
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let botText = "";
      
      const botMsgId = Date.now() + 1;
      // Add empty message placeholder
      setMessages(prev => [...prev, {
        id: botMsgId,
        text: "",
        sender: 'bot',
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        type: 'ollama',
        userQuery: text,
        answerId: null // Will be updated from stream
      }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        botText += chunk;

        if (botText.includes('|||')) {
          const parts = botText.split('|||');
          const cleanText = parts[0];
          const ansId = parts[1];
          const suggestions = parts[2]?.split('###').filter(s => s.trim()) || [];
          
          setMessages(prev => prev.map(m => 
            m.id === botMsgId ? { ...m, text: cleanText, suggestions: suggestions, answerId: ansId } : m
          ));
        } else {
          setMessages(prev => prev.map(m => 
            m.id === botMsgId ? { ...m, text: botText } : m
          ));
        }
      }

    } catch (error) {
      console.error("Chat Error:", error);
      // ... existing error handling
    } finally {
      setIsLoading(false);
    }
  };

  const handleFeedback = async (msgId, rating) => {
    const msg = messages.find(m => m.id === msgId);
    if (!msg || !msg.userQuery) return;

    try {
      await fetch('http://127.0.0.1:5000/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: msg.userQuery,
          response: msg.text,
          rating: rating,
          answer_id: msg.answerId
        })
      });

      // Mark as rated and store the rating value
      setMessages(prev => prev.map(m => 
        m.id === msgId ? { ...m, rated: true, userRating: rating } : m
      ));
    } catch (e) {
      console.error("Feedback failed", e);
    }
  };

  const handleReset = async () => {
    try {
      await fetch('http://127.0.0.1:5000/reset', { method: 'POST' });
      setMessages([{
        id: Date.now(),
        text: "Chat history cleared. How can I help you now?",
        sender: 'bot',
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }]);
    } catch (e) {
      console.error("Reset failed", e);
    }
  };

  return (
    <div className="app-wrapper">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="logo-section">
          <div className="bot-avatar" style={{ width: '32px', height: '32px' }}>
            <Bot size={18} />
          </div>
          <h2>MITS AI</h2>
        </div>

        <nav className="nav-section">
          <span className="nav-title">Quick Links</span>
          <button 
            onClick={() => navigate('/')} 
            className={`nav-item ${view === 'chat' ? 'active' : ''}`}
          >
            <MessageSquare size={18} /> Chat
          </button>
          <button 
            onClick={() => navigate('/admin')} 
            className={`nav-item ${view === 'admin' ? 'active' : ''}`}
          >
            <Info size={18} /> Admin Dashboard
          </button>
          <a href="http://moodle.mitsgwalior.in/" target="_blank" className="nav-item">
            <Globe size={18} /> Moodle (Old)
          </a>
          <a href="https://moodle.mitseb.in/" target="_blank" className="nav-item">
            <Globe size={18} /> Moodle (2024+)
          </a>
          <a href="https://mitsims.in/" target="_blank" className="nav-item">
            <MessageSquare size={18} /> IMS Portal
          </a>
          <a href="https://mitsgwalior.in/" target="_blank" className="nav-item">
            <Globe size={18} /> MITS Website
          </a>
        </nav>

        <nav className="nav-section">
          <span className="nav-title">Resources</span>
          <a href="#" className="nav-item">
            <GraduationCap size={18} /> Registration Guide
          </a>
          <a href="#" className="nav-item">
            <BookOpen size={18} /> Documentation
          </a>
        </nav>

        <div className="status-card">
          <div className="status-item">
            <div className="status-dot online" />
            <span>System Online</span>
          </div>
          <div className="status-item" style={{ marginTop: '8px', opacity: 0.7 }}>
            <Bot size={14} />
            <span>Llama 3.2 Active</span>
          </div>
        </div>
      </aside>

      {/* Main Chat Container */}
      {/* Main Container */}
      <div className="app-container">
        {view === 'chat' ? (
          <>
            <header className="app-header">
              <div className="bot-avatar">
                <Bot size={22} />
              </div>
              <div className="header-info">
                <h1>MITS Assistant</h1>
                <p>Powered by local LLM • Knowledge Base 2.0</p>
              </div>
              <button 
                onClick={handleReset}
                style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}
                title="Reset Chat"
              >
                <RefreshCw size={18} />
              </button>
            </header>

            <main className="messages-area">
              <AnimatePresence initial={false}>
                {messages.map((msg) => (
                  <motion.div 
                    key={msg.id}
                    initial={{ opacity: 0, y: 10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    className={`message-row ${msg.sender}`}
                  >
                    <div className="message-bubble">
                      <div className="markdown-content">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {msg.text}
                        </ReactMarkdown>
                      </div>
                      <div className="message-footer">
                        <span className="message-time">{msg.time}</span>
                        {msg.sender === 'bot' && msg.userQuery && (
                      <div className="rating-area">
                        {!msg.rated ? (
                          <div className="rating-container">
                            {[1, 2, 3, 4, 5].map(star => (
                              <button 
                                key={star} 
                                onClick={() => handleFeedback(msg.id, star)}
                                className="star-btn"
                              >
                                <Star size={12} />
                              </button>
                            ))}
                          </div>
                        ) : (
                          <div className="submitted-rating">
                            {[1, 2, 3, 4, 5].map(star => (
                              <Star 
                                key={star} 
                                size={12} 
                                fill={star <= msg.userRating ? "#fbbf24" : "transparent"} 
                                color={star <= msg.userRating ? "#fbbf24" : "var(--text-muted)"} 
                              />
                            ))}
                            <span className="thanks-text">Thanks!</span>
                          </div>
                        )}
                      </div>
                    )}
                      </div>
                      
                      {msg.suggestions && msg.suggestions.length > 0 && (
                        <div className="suggestions-container">
                          {msg.suggestions.map((s, i) => (
                            <button 
                              key={i} 
                              className="suggestion-chip"
                              onClick={() => handleSend(s)}
                            >
                              {s}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
              
              {isLoading && (
                <motion.div 
                  initial={{ opacity: 0 }} 
                  animate={{ opacity: 1 }}
                  className="message-row bot"
                >
                  <div className="message-bubble" style={{ padding: '16px 24px' }}>
                    <div className="typing">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </motion.div>
              )}
              <div ref={messagesEndRef} />
            </main>

            <footer className="input-area">
              <form 
                className="input-container" 
                onSubmit={(e) => { e.preventDefault(); handleSend(); }}
              >
                <input 
                  type="text" 
                  placeholder="Ask anything about MITS..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                />
                <button 
                  type="button"
                  className={`voice-btn ${isListening ? 'active' : ''}`}
                  onClick={handleVoiceInput}
                >
                  {isListening ? <MicOff size={18} /> : <Mic size={18} />}
                </button>
                <button 
                  type="submit" 
                  className="send-btn"
                  disabled={!input.trim() || isLoading}
                >
                  <Send size={18} />
                </button>
              </form>
            </footer>
          </>
        ) : (
          <div className="admin-view">
            <header className="app-header">
              <div className="bot-avatar" style={{ background: 'var(--accent)' }}>
                <Star size={22} />
              </div>
              <div className="header-info">
                <h1>Feedback Analytics</h1>
                <p>Monitor student queries and bot performance</p>
              </div>
              <div className="admin-actions">
                <button 
                  className="refresh-btn-admin" 
                  onClick={fetchAdminData}
                  title="Refresh Data"
                >
                  <RefreshCw size={18} />
                </button>
              </div>
              <div className="admin-tabs">
                <button 
                  className={`tab-btn ${adminTab === 'feedback' ? 'active' : ''}`}
                  onClick={() => setAdminTab('feedback')}
                >
                  Feedbacks
                </button>
                <button 
                  className={`tab-btn ${adminTab === 'learning' ? 'active' : ''}`}
                  onClick={() => setAdminTab('learning')}
                >
                  AI Learning
                </button>
              </div>
            </header>
            
            <div className="admin-content">
              {adminTab === 'feedback' ? (
                <div className="table-wrapper">
                  <table className="admin-table">
                    <thead>
                      <tr>
                        <th>Time</th>
                        <th>User Query</th>
                        <th>Response</th>
                        <th>Rating</th>
                      </tr>
                    </thead>
                    <tbody>
                      {adminData.map((f) => (
                        <tr key={f.id}>
                          <td>{formatDate(f.timestamp)}</td>
                          <td className="query-cell">{f.query}</td>
                          <td className="resp-cell">{String(f.response || "").substring(0, 100)}...</td>
                          <td>
                            <div className="rating-stars">
                              {[...Array(f.rating)].map((_, i) => (
                                <Star key={i} size={12} fill="#fbbf24" color="#fbbf24" />
                              ))}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="table-wrapper">
                  <table className="admin-table">
                    <thead>
                      <tr>
                        <th>FAQ ID</th>
                        <th>Avg Rating</th>
                        <th>Total Feedback</th>
                        <th>RL Impact</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {adminStats.map((s) => (
                        <tr key={s.answer_id || 'general'}>
                          <td className="query-cell">
                            {s.answer_id ? `FAQ ID: ${s.answer_id}` : "General Training Data"}
                          </td>
                          <td>
                            <div className="rating-pill">
                              {s.avg_rating} <Star size={10} fill="#fbbf24" color="#fbbf24" style={{marginLeft: 2}} />
                            </div>
                          </td>
                          <td>{s.total_feedback} Feedbacks</td>
                          <td className={s.rl_impact > 0 ? "impact-reward" : (s.rl_impact < 0 ? "impact-penalty" : "")}>
                            {s.rl_impact > 0 ? `+${s.rl_impact} Reward` : (s.rl_impact < 0 ? `${s.rl_impact} Penalty` : "0")}
                          </td>
                          <td>
                            <span className={`status-pill ${s.status?.toLowerCase().includes('rewarded') ? 'boost' : (s.status?.toLowerCase().includes('penalized') ? 'penalty' : '')}`}>
                              {s.status || "Normal"}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
