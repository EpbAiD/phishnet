#!/bin/bash
#
# PhishNet EC2 Daemon Setup Script
# =================================
# Sets up the 24/7 continuous collector as a systemd service on EC2
#
# Usage:
#   chmod +x scripts/setup_ec2_daemon.sh
#   sudo ./scripts/setup_ec2_daemon.sh
#

set -e

echo "=============================================="
echo "PhishNet Continuous Collector - EC2 Setup"
echo "=============================================="

# Configuration
INSTALL_DIR="/home/ec2-user/phishnet"
SERVICE_NAME="phishnet-collector"
LOG_DIR="/var/log/phishnet"
PYTHON_BIN="/usr/bin/python3.8"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)"
   exit 1
fi

# Create log directory
echo "Creating log directory..."
mkdir -p $LOG_DIR
chown ec2-user:ec2-user $LOG_DIR

# Install Python dependencies if needed
echo "Installing Python dependencies..."
if [ -f "$INSTALL_DIR/requirements.txt" ]; then
    sudo -u ec2-user $PYTHON_BIN -m pip install -r $INSTALL_DIR/requirements.txt --quiet
fi

# Install additional required packages
sudo -u ec2-user $PYTHON_BIN -m pip install boto3 pandas requests dnspython python-whois ipwhois --quiet

# Create systemd service file
echo "Creating systemd service..."
cat > /etc/systemd/system/${SERVICE_NAME}.service << EOF
[Unit]
Description=PhishNet Continuous URL Collector
Documentation=https://github.com/EpbAiD/phishnet
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ec2-user
Group=ec2-user
WorkingDirectory=$INSTALL_DIR
Environment="PYTHONPATH=$INSTALL_DIR"
Environment="AWS_REGION=us-east-1"
Environment="S3_BUCKET=phishnet-data"
Environment="COLLECTION_INTERVAL=900"
Environment="MAX_URLS_PER_CYCLE=1000"
Environment="LOG_FILE=$LOG_DIR/collector.log"

ExecStart=$PYTHON_BIN $INSTALL_DIR/scripts/continuous_collector_daemon.py
ExecStop=/bin/kill -SIGTERM \$MAINPID

# Restart policy
Restart=always
RestartSec=60
StartLimitBurst=5
StartLimitIntervalSec=300

# Resource limits
MemoryLimit=1G
CPUQuota=80%

# Security
NoNewPrivileges=true
PrivateTmp=true

# Logging
StandardOutput=append:$LOG_DIR/collector.log
StandardError=append:$LOG_DIR/collector.log

[Install]
WantedBy=multi-user.target
EOF

# Create logrotate configuration
echo "Configuring log rotation..."
cat > /etc/logrotate.d/${SERVICE_NAME} << EOF
$LOG_DIR/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 ec2-user ec2-user
    postrotate
        systemctl reload ${SERVICE_NAME} > /dev/null 2>&1 || true
    endscript
}
EOF

# Reload systemd
echo "Reloading systemd..."
systemctl daemon-reload

# Enable service to start on boot
echo "Enabling service..."
systemctl enable ${SERVICE_NAME}

# Start the service
echo "Starting service..."
systemctl start ${SERVICE_NAME}

# Wait a moment for startup
sleep 3

# Check status
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Service Status:"
systemctl status ${SERVICE_NAME} --no-pager -l

echo ""
echo "=============================================="
echo "Useful Commands:"
echo "=============================================="
echo "  View logs:          sudo journalctl -u ${SERVICE_NAME} -f"
echo "  View log file:      tail -f $LOG_DIR/collector.log"
echo "  Check status:       sudo systemctl status ${SERVICE_NAME}"
echo "  Stop service:       sudo systemctl stop ${SERVICE_NAME}"
echo "  Start service:      sudo systemctl start ${SERVICE_NAME}"
echo "  Restart service:    sudo systemctl restart ${SERVICE_NAME}"
echo "  Disable autostart:  sudo systemctl disable ${SERVICE_NAME}"
echo ""
echo "The collector will run every 15 minutes, collecting URLs from"
echo "13 threat intelligence feeds and extracting URL/DNS/WHOIS features."
echo ""
