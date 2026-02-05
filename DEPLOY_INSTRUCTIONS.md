# Hướng dẫn Deploy lên Server

## Thông tin Server
- **IP**: 152.42.252.101
- **Domain**: autoscaling.me
- **Email**: dat2801zz@gmail.com
- **User**: dataflow

## Các bước thực hiện:

### 1. Kết nối SSH vào server
```bash
ssh dataflow@152.42.252.101
```

### 2. Dọn dẹp code cũ và container cũ
```bash
# Vào thư mục dự án (nếu có)
cd ~/datafollow

# Dừng tất cả container
docker-compose down
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)

# Xóa code cũ
cd ~
rm -rf datafollow
mkdir -p datafollow
```

### 3. Upload code mới từ máy local

**Cách 1: Dùng script tự động** (Khuyến nghị)

Mở Terminal mới trên **máy local** (Windows):
```powershell
cd C:\Users\Admin\OneDrive\Documents\python\datafollow
.\upload-to-server.ps1
```

**Cách 2: Upload thủ công**
```powershell
cd C:\Users\Admin\OneDrive\Documents\python\datafollow
scp -r app digitalocean models src Dockerfile.api Dockerfile.dashboard requirements.txt .dockerignore dataflow@152.42.252.101:~/datafollow/
```

### 4. Quay lại SSH trên server và deploy

```bash
# Vào thư mục dự án
cd ~/datafollow

# Build và chạy containers
cd digitalocean
docker-compose -f docker-compose.droplet.yml down
docker-compose -f docker-compose.droplet.yml up --build -d

# Kiểm tra trạng thái
docker-compose -f docker-compose.droplet.yml ps

# Xem logs
docker-compose -f docker-compose.droplet.yml logs -f
```

### 5. Cấu hình SSL (nếu chưa có)

```bash
# Lấy SSL certificate
docker run --rm \
    -v ~/datafollow/digitalocean/certbot/conf:/etc/letsencrypt \
    -v ~/datafollow/digitalocean/certbot/www:/var/www/certbot \
    certbot/certbot certonly \
    --standalone \
    --email dat2801zz@gmail.com \
    --agree-tos \
    --no-eff-email \
    -d autoscaling.me
```

### 6. Kiểm tra kết quả

- API: https://autoscaling.me/api
- Dashboard: https://autoscaling.me

## Lệnh hữu ích

```bash
# Xem logs
docker-compose -f docker-compose.droplet.yml logs -f api
docker-compose -f docker-compose.droplet.yml logs -f dashboard

# Restart services
docker-compose -f docker-compose.droplet.yml restart

# Stop all
docker-compose -f docker-compose.droplet.yml down

# Rebuild
docker-compose -f docker-compose.droplet.yml up --build -d
```
