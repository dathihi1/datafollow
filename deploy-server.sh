#!/bin/bash
# Deploy script - Run this on the server
# Usage: bash deploy-server.sh

set -e

APP_DIR="/root/datafollow"
DOMAIN="autoscaling.me"
EMAIL="dat2801zz@gmail.com"

echo "================================================"
echo "Deploying Autoscaling Application"
echo "================================================"

# Step 1: Stop and remove old containers
echo ""
echo "Step 1: Stopping old containers..."
cd $APP_DIR 2>/dev/null || true
docker-compose down 2>/dev/null || true
docker stop $(docker ps -aq) 2>/dev/null || true
docker rm $(docker ps -aq) 2>/dev/null || true

# Step 2: Clean up old code
echo ""
echo "Step 2: Cleaning old code..."
cd /root
rm -rf $APP_DIR
mkdir -p $APP_DIR
cd $APP_DIR

echo "✅ Old code removed"

# Step 3: You will upload new code here
echo ""
echo "Step 3: Ready for new code upload"
echo "Now upload your code to: $APP_DIR"
echo ""
echo "After uploading, run the following commands:"
echo "cd $APP_DIR"
echo "docker-compose -f docker-compose.droplet.yml up --build -d"
echo ""
echo "================================================"
