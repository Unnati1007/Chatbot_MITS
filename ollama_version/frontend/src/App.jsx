import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, RefreshCw, MessageSquare, Globe, BookOpen, GraduationCap, Mic, MicOff } from 'lucide-react';
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
  const messagesEndRef = useRef(null);

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
        type: 'ollama'
      }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        botText += decoder.decode(value, { stream: true });
        
        // Update the last message text
        setMessages(prev => prev.map(m => 
          m.id === botMsgId ? { ...m, text: botText } : m
        ));
      }

    } catch (error) {
      console.error("Chat Error:", error);
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        text: "**Connection failed.** Please ensure the backend is running and Ollama is active.",
        sender: 'bot',
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        type: 'error'
      }]);
    } finally {
      setIsLoading(false);
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
          <a href="#" className="nav-item active">
            <MessageSquare size={18} /> Chat
          </a>
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
      <div className="app-container">
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
                  <span className="message-time">{msg.time}</span>
                  
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
              title={isListening ? "Stop listening" : "Voice input"}
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
      </div>
    </div>
  );
}

export default App;
