import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, RefreshCw, MessageSquare } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_URL = 'http://127.0.0.1:5000/chat';

function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! 👋 I'm your MITS AI Assistant. Ask me anything about Moodle, IMS, or Registration.",
      sender: 'bot',
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      type: 'rule'
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

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
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text })
      });

      const data = await response.json();

      const botMsg = {
        id: Date.now() + 1,
        text: data.reply || "Sorry, I'm having trouble connecting.",
        sender: 'bot',
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        type: data.type,
        suggestions: data.suggestions || []
      };

      setMessages(prev => [...prev, botMsg]);
    } catch (error) {
      console.error("Chat Error:", error);
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        text: "Connection failed. Please ensure the backend is running.",
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
    <div className="app-container">
      <header className="app-header">
        <div className="bot-avatar">
          <Bot size={24} />
        </div>
        <div className="header-info">
          <h1>MITS Assistant</h1>
          <p><span className="status-dot"></span> Online • AI Powered</p>
        </div>
        <button 
          onClick={handleReset}
          style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}
          title="Reset Chat"
        >
          <RefreshCw size={20} />
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
                <div className="message-content">{msg.text}</div>
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
            <div className="message-bubble" style={{ padding: '12px 20px' }}>
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
          <MessageSquare size={20} color="var(--text-muted)" />
          <input 
            type="text" 
            placeholder="Type your message here..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
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
  );
}

export default App;
