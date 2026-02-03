import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { APP_NAME, APP_DESCRIPTION } from './config';
import { checkHealth } from './api';
import Scanner from './components/Scanner';
import Dashboard from './components/Dashboard';
import './App.css';

function Navigation() {
  const location = useLocation();

  return (
    <nav className="nav">
      <Link
        to="/"
        className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}
      >
        Scanner
      </Link>
      <Link
        to="/dashboard"
        className={`nav-link ${location.pathname === '/dashboard' ? 'active' : ''}`}
      >
        Dashboard
      </Link>
    </nav>
  );
}

function App() {
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  useEffect(() => {
    checkHealth()
      .then(() => setApiStatus('online'))
      .catch(() => setApiStatus('offline'));
  }, []);

  return (
    <Router>
      <div className="app">
        <header className="header">
          <h1>{APP_NAME}</h1>
          <p>{APP_DESCRIPTION}</p>
          <div className={`api-status ${apiStatus}`}>
            API: {apiStatus === 'checking' ? 'Checking...' : apiStatus === 'online' ? 'Online' : 'Offline'}
          </div>
          <Navigation />
        </header>

        <main className="main">
          <Routes>
            <Route path="/" element={<Scanner />} />
            <Route path="/dashboard" element={<Dashboard />} />
          </Routes>
        </main>

        <footer className="footer">
          <p>PhishNet - Protecting users from phishing attacks</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;
