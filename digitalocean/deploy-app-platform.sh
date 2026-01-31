#!/bin/bash

# DigitalOcean App Platform Deployment Script
# Uses doctl CLI to deploy app

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <app-name> <github-repo>"
    echo "Example: $0 autoscaling-analysis your-username/autoscaling-analysis"
    exit 1
fi

APP_NAME=$1
GITHUB_REPO=$2

echo "================================================"
echo "DigitalOcean App Platform Deployment"
echo "================================================"

# Check if doctl is installed
if ! command -v doctl &> /dev/null; then
    echo "❌ doctl CLI is not installed"
    echo "Install it from: https://docs.digitalocean.com/reference/doctl/how-to/install/"
    exit 1
fi

# Check authentication
echo "Checking authentication..."
if ! doctl account get &> /dev/null; then
    echo "❌ Not authenticated with DigitalOcean"
    echo "Run: doctl auth init"
    exit 1
fi

echo "✅ Authentication verified"

# Update app.yaml with actual values
echo ""
echo "Updating app.yaml configuration..."
sed -i "s/your-username\/autoscaling-analysis/$GITHUB_REPO/g" app.yaml

# Create or update app
echo ""
echo "Deploying to App Platform..."

# Check if app exists
if doctl apps list | grep -q "$APP_NAME"; then
    echo "App exists, updating..."
    APP_ID=$(doctl apps list --format ID,Spec.Name --no-header | grep "$APP_NAME" | awk '{print $1}')
    doctl apps update $APP_ID --spec app.yaml
else
    echo "Creating new app..."
    doctl apps create --spec app.yaml
fi

echo ""
echo "================================================"
echo "✅ Deployment initiated!"
echo "================================================"
echo ""
echo "Check deployment status:"
echo "  doctl apps list"
echo ""
echo "View app details:"
echo "  doctl apps get <app-id>"
echo ""
echo "View logs:"
echo "  doctl apps logs <app-id> --type build"
echo "  doctl apps logs <app-id> --type run"
echo ""
echo "================================================"
