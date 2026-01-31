# DigitalOcean Droplet Deployment Script (PowerShell)
# This runs on your LOCAL machine to deploy to the droplet
# Usage: .\deploy-droplet.ps1 -DropletIP "your.droplet.ip" -Domain "yourdomain.com" -Email "your@email.com"

param(
    [Parameter(Mandatory=$true)]
    [string]$DropletIP,
    
    [Parameter(Mandatory=$true)]
    [string]$Domain,
    
    [Parameter(Mandatory=$true)]
    [string]$Email,
    
    [string]$SSHUser = "root"
)

$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "DigitalOcean Droplet Deployment" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Droplet IP: $DropletIP"
Write-Host "Domain: $Domain"
Write-Host "Email: $Email"
Write-Host "SSH User: $SSHUser"
Write-Host "================================================" -ForegroundColor Cyan

# Step 1: Check SSH connection
Write-Host "`nStep 1: Testing SSH connection..." -ForegroundColor Yellow
try {
    ssh -o BatchMode=yes -o ConnectTimeout=5 ${SSHUser}@${DropletIP} "echo 'SSH connection successful'" 2>$null
    Write-Host "✅ SSH connection working" -ForegroundColor Green
} catch {
    Write-Host "❌ Cannot connect via SSH. Please check:" -ForegroundColor Red
    Write-Host "  1. Droplet IP is correct: $DropletIP"
    Write-Host "  2. SSH key is configured"
    Write-Host "  3. Firewall allows SSH (port 22)"
    exit 1
}

# Step 2: Copy files to droplet
Write-Host "`nStep 2: Copying files to droplet..." -ForegroundColor Yellow

# Create deployment directory
ssh ${SSHUser}@${DropletIP} "mkdir -p /opt/autoscaling-analysis"

# Copy entire project
Write-Host "Copying project files..."
scp -r ../* ${SSHUser}@${DropletIP}:/opt/autoscaling-analysis/

Write-Host "✅ Files copied successfully" -ForegroundColor Green

# Step 3: Update configuration with domain
Write-Host "`nStep 3: Updating configuration..." -ForegroundColor Yellow
$nginxConfig = @"
sed -i 's/yourdomain.com/$Domain/g' /opt/autoscaling-analysis/digitalocean/nginx/conf.d/app.conf
"@

ssh ${SSHUser}@${DropletIP} $nginxConfig
Write-Host "✅ Configuration updated" -ForegroundColor Green

# Step 4: Run setup script on droplet
Write-Host "`nStep 4: Running setup script on droplet..." -ForegroundColor Yellow
Write-Host "This may take 5-10 minutes..." -ForegroundColor Yellow

$setupScript = @"
cd /opt/autoscaling-analysis/digitalocean
chmod +x deploy-droplet.sh
./deploy-droplet.sh $Domain $Email
"@

ssh ${SSHUser}@${DropletIP} $setupScript

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "✅ Deployment completed!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Your application should be accessible at:"
Write-Host "  🌐 https://$Domain" -ForegroundColor Yellow
Write-Host ""
Write-Host "Note: DNS propagation may take up to 48 hours."
Write-Host "Make sure your domain's DNS points to: $DropletIP"
Write-Host ""
Write-Host "To check status:"
Write-Host "  ssh ${SSHUser}@${DropletIP}" -ForegroundColor Cyan
Write-Host "  cd /opt/autoscaling-analysis/digitalocean" -ForegroundColor Cyan
Write-Host "  sudo docker-compose -f docker-compose.droplet.yml ps" -ForegroundColor Cyan
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
