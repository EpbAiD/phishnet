import React from 'react';
import './Dashboard.css';

function Dashboard() {
  return (
    <div className="dashboard">
      <div className="coming-soon">
        <h2>Dashboard</h2>
        <p>Coming Soon</p>
        <div className="planned-features">
          <h3>Planned Features:</h3>
          <ul>
            <li>Scan history and statistics</li>
            <li>Model performance metrics</li>
            <li>Feedback analytics</li>
            <li>Threat intelligence feed</li>
            <li>API usage monitoring</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
