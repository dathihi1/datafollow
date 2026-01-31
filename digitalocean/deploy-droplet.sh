#!/bin/bash

# DigitalOcean Droplet Deployment Script
# Usage: ./deploy-droplet.sh yourdomain.com your_email@example.com

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <domain> <email>"
    echo "Example: $0 yourdomain.com admin@yourdomain.com"
    exit 1
fi

DOMAIN=$1
EMAIL=$2

echo "================================================"
echo "DigitalOcean Droplet Setup"
echo "================================================"
echo "Domain: $DOMAIN"
echo "Email: $EMAIL"
echo "================================================"

# Step 1: Update system
echo ""
echo "Step 1: Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Step 2: Install Docker
echo ""
echo "Step 2: Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "✅ Docker installed"
else
    echo "✅ Docker already installed"
fi

# Step 3: Install Docker Compose
echo ""
echo "Step 3: Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "✅ Docker Compose installed"
else
    echo "✅ Docker Compose already installed"
fi

# Step 4: Install additional tools
echo ""
echo "Step 4: Installing additional tools..."
sudo apt-get install -y git curl nginx-extras certbot python3-certbot-nginx

# Step 5: Clone repository (if not exists)
echo ""
echo "Step 5: Setting up application directory..."
APP_DIR="/opt/autoscaling-analysis"
if [ ! -d "$APP_DIR" ]; then
    echo "Creating application directory..."
    sudo mkdir -p $APP_DIR
    sudo chown $USER:$USER $APP_DIR
else
    echo "✅ Application directory exists"
fi

# Step 6: Update Nginx configuration with domain
echo ""
echo "Step 6: Configuring Nginx..."
cd digitalocean
sed "s/yourdomain.com/$DOMAIN/g" nginx/conf.d/app.conf > /tmp/app.conf
sudo cp /tmp/app.conf nginx/conf.d/app.conf

# Step 7: Get SSL certificate
echo ""
echo "Step 7: Obtaining SSL certificate..."
# First, start nginx temporarily for certbot
sudo docker-compose -f docker-compose.droplet.yml up -d nginx

# Wait for nginx to start
sleep 5

# Get certificate
sudo docker run --rm \
    -v $(pwd)/certbot/conf:/etc/letsencrypt \
    -v $(pwd)/certbot/www:/var/www/certbot \
    certbot/certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email $EMAIL \
    --agree-tos \
    --no-eff-email \
    -d $DOMAIN \
    -d www.$DOMAIN

echo "✅ SSL certificate obtained"

# Step 8: Build and start all services
echo ""
echo "Step 8: Building and starting services..."
sudo docker-compose -f docker-compose.droplet.yml down
sudo docker-compose -f docker-compose.droplet.yml up -d --build

# Step 9: Wait for services to be healthy
echo ""
echo "Step 9: Waiting for services to be ready..."
sleep 20

# Check services
echo "Checking API health..."
for i in {1..30}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API is healthy"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ API health check failed"
        sudo docker-compose -f docker-compose.droplet.yml logs api
        exit 1
    fi
    sleep 2
done

echo "Checking Dashboard health..."
for i in {1..30}; do
    if curl -sf http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        echo "✅ Dashboard is healthy"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Dashboard health check failed"
        sudo docker-compose -f docker-compose.droplet.yml logs dashboard
        exit 1
    fi
    sleep 2
done

# Step 10: Setup auto-renewal for SSL
echo ""
echo "Step 10: Setting up SSL auto-renewal..."
# Certbot container already handles renewal

# Step 11: Setup monitoring (optional)
echo ""
echo "Step 11: Setting up basic monitoring..."
# Install netdata for monitoring
# bash <(curl -Ss https://my-netdata.io/kickstart.sh) --dont-wait

echo ""
echo "================================================"
echo "✅ Deployment completed successfully!"
echo "================================================"
echo ""
echo "Your application is now running at:"
echo "  🌐 https://$DOMAIN"
echo "  📊 Dashboard: https://$DOMAIN"
echo "  🔌 API Docs: https://$DOMAIN/docs"
echo ""
echo "Useful commands:"
echo "  View logs: cd $APP_DIR/digitalocean && sudo docker-compose -f docker-compose.droplet.yml logs -f"
echo "  Restart: cd $APP_DIR/digitalocean && sudo docker-compose -f docker-compose.droplet.yml restart"
echo "  Stop: cd $APP_DIR/digitalocean && sudo docker-compose -f docker-compose.droplet.yml down"
echo "  Update: git pull && sudo docker-compose -f docker-compose.droplet.yml up -d --build"
echo ""
echo "================================================"
