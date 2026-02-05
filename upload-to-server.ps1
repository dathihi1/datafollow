# Script upload code lên server
# Usage: .\upload-to-server.ps1

$SERVER_IP = "152.42.252.101"
$SERVER_USER = "dataflow"
$SERVER_PATH = "~/datafollow"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Upload Code to Server" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Server: $SERVER_USER@$SERVER_IP"
Write-Host "Path: $SERVER_PATH"
Write-Host "================================================" -ForegroundColor Cyan

# Danh sách file/folder cần upload
$items = @(
    "app",
    "digitalocean", 
    "models",
    "src",
    "Dockerfile.api",
    "Dockerfile.dashboard",
    "requirements.txt",
    ".dockerignore"
)

Write-Host "`nUploading files..." -ForegroundColor Yellow

foreach ($item in $items) {
    if (Test-Path $item) {
        Write-Host "  - Uploading $item..." -ForegroundColor Gray
        scp -r $item "${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/"
    } else {
        Write-Host "  - Skipping $item (not found)" -ForegroundColor DarkGray
    }
}

Write-Host "`n✅ Upload completed!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. SSH to server: ssh $SERVER_USER@$SERVER_IP" -ForegroundColor Cyan
Write-Host "  2. Deploy: cd ~/datafollow/digitalocean && docker-compose -f docker-compose.droplet.yml up --build -d" -ForegroundColor Cyan
Write-Host ""
