# Hướng dẫn Deploy lên DigitalOcean Droplet - Chi tiết từng bước

> **Phương pháp này rẻ nhất**: $6-12/tháng so với $15/tháng của App Platform

## 📋 Chuẩn bị (5 phút)

### Bạn cần có:
- ✅ Tài khoản DigitalOcean
- ✅ Tên miền đã mua (ví dụ: yourdomain.com)
- ✅ Máy tính có PowerShell (Windows) hoặc Terminal (Mac/Linux)
- ✅ Code project này

### Công cụ cần cài:
```powershell
# Kiểm tra Git đã cài chưa
git --version

# Nếu chưa có, download tại: https://git-scm.com/
```

---

## 🚀 BƯỚC 1: Tạo Droplet (5 phút)

### 1.1. Đăng ký DigitalOcean (nếu chưa có)

1. Vào: https://m.do.co/c/your-referral (nhận $200 credit)
2. Đăng ký với email
3. Xác thực email
4. Thêm thẻ tín dụng (sẽ không charge nếu còn credit)

### 1.2. Tạo Droplet

1. **Login** vào DigitalOcean Dashboard
2. Click **Create** → **Droplets**
3. **Chọn image**:
   - Choose an image: **Ubuntu 22.04 (LTS) x64**
   
4. **Chọn plan**:
   - Droplet Type: **Basic**
   - CPU options: **Regular Intel với SSD**
   - Size: 
     - `$6/month` (1GB RAM) - Cho dev/test
     - `$12/month` (2GB RAM) - **KHUYẾN NGHỊ** cho production
     - `$18/month` (2GB RAM, 2vCPU) - Nếu cần performance tốt hơn

5. **Chọn datacenter**:
   - Region: **Singapore** (gần Việt Nam nhất, ping thấp)
   
6. **Authentication** (QUAN TRỌNG):
   
   **Cách 1: SSH Key (KHUYẾN NGHỊ - An toàn hơn)**
   ```powershell
   # Mở PowerShell, tạo SSH key
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
   
   # Nhấn Enter 3 lần (dùng default settings)
   
   # Xem public key
   type $env:USERPROFILE\.ssh\id_rsa.pub
   ```
   
   Copy toàn bộ nội dung, paste vào DigitalOcean:
   - Click **New SSH Key**
   - Paste key vào
   - Đặt tên: "My-Windows-PC"
   - Click **Add SSH Key**
   
   **Cách 2: Password (Đơn giản nhưng kém bảo mật)**
   - Chọn "Password"
   - Nhập password mạnh (ít nhất 12 ký tự)

7. **Finalize Details**:
   - Quantity: **1 Droplet**
   - Hostname: `autoscaling-server` (hoặc tên bạn thích)
   - Tags: `production` (optional)
   - Backups: Không chọn (tiết kiệm, có thể bật sau)

8. Click **Create Droplet**

9. **Chờ 1-2 phút**, Droplet sẽ sẵn sàng

10. **Copy Droplet IP**: 
    - Ví dụ: `165.227.xxx.xxx`
    - Lưu lại IP này!

---

## 🌐 BƯỚC 2: Cấu hình DNS (10-30 phút)

> **Quan trọng**: Làm bước này TRƯỚC khi deploy để DNS kịp propagate

### 2.1. Nếu domain ở GoDaddy, Namecheap, etc.

1. **Login** vào tài khoản domain provider
2. Tìm **DNS Management** hoặc **DNS Settings**
3. **Xóa** các A records cũ (nếu có)
4. **Thêm** DNS records mới:

```
Type    Host/Name    Value/Points to      TTL
----    ---------    ------------------   ----
A       @            YOUR_DROPLET_IP      3600
A       www          YOUR_DROPLET_IP      3600
```

**Ví dụ cụ thể:**
```
Type    Host    Value             TTL
A       @       165.227.xxx.xxx   3600
A       www     165.227.xxx.xxx   3600
```

5. **Save changes**

### 2.2. Hoặc dùng DigitalOcean Nameservers (Khuyến nghị)

**Ưu điểm**: Quản lý tập trung, DNS nhanh hơn

1. Vào DigitalOcean → **Networking** → **Domains**
2. Click **Add Domain**
3. Nhập: `yourdomain.com` → Click **Add Domain**
4. DigitalOcean sẽ show 3 nameservers:
   ```
   ns1.digitalocean.com
   ns2.digitalocean.com
   ns3.digitalocean.com
   ```
5. Copy 3 nameservers này
6. Vào domain provider → **Nameservers** hoặc **DNS**
7. Thay thế nameservers cũ bằng 3 nameservers của DO
8. **Save**
9. Quay lại DigitalOcean → **Add Record**:
   ```
   Type: A
   Hostname: @
   Will Direct to: <chọn droplet của bạn>
   TTL: 3600
   ```
10. Thêm record thứ 2:
    ```
    Type: A
    Hostname: www
    Will Direct to: <chọn droplet của bạn>
    TTL: 3600
    ```

### 2.3. Kiểm tra DNS (sau 5-30 phút)

```powershell
# Kiểm tra DNS đã trỏ đúng chưa
nslookup yourdomain.com

# Kết quả mong đợi:
# Address: YOUR_DROPLET_IP
```

**Nếu chưa thấy IP**: Chờ thêm 10-30 phút (DNS propagation)

---

## 💻 BƯỚC 3: Deploy Application (10-15 phút)

### 3.1. Test SSH Connection

```powershell
# Thay YOUR_DROPLET_IP bằng IP thực tế
ssh root@YOUR_DROPLET_IP

# Nếu dùng SSH key, sẽ connect ngay
# Nếu dùng password, nhập password bạn đã tạo

# Nếu hỏi "Are you sure...?", gõ: yes
```

**Nếu connect thành công**, bạn sẽ thấy:
```
Welcome to Ubuntu 22.04.x LTS
root@autoscaling-server:~#
```

**Nếu không connect được:**
- Kiểm tra IP đúng chưa
- Kiểm tra firewall local (tắt VPN nếu có)
- Thử dùng password thay vì SSH key

Gõ `exit` để thoát, quay về máy local.

### 3.2. Chuẩn bị files

```powershell
# Mở PowerShell tại thư mục project
cd c:\Users\Admin\OneDrive\Documents\python\datafollow

# Kiểm tra các file cần thiết có đủ không
dir digitalocean

# Bạn cần thấy:
# - deploy-droplet.ps1
# - docker-compose.droplet.yml
# - nginx/
```

### 3.3. Chạy Deploy Script

```powershell
# Deploy một lệnh (thay thông tin thực tế)
.\digitalocean\deploy-droplet.ps1 `
    -DropletIP "165.227.xxx.xxx" `
    -Domain "yourdomain.com" `
    -Email "your@email.com"

# Giải thích:
# -DropletIP: IP của droplet bạn vừa tạo
# -Domain: Tên miền của bạn (KHÔNG có http:// hoặc www)
# -Email: Email của bạn (để nhận thông báo SSL)
```

**Script sẽ tự động:**
1. ✅ Test SSH connection
2. ✅ Copy toàn bộ files lên server
3. ✅ Install Docker & Docker Compose
4. ✅ Setup Nginx reverse proxy
5. ✅ Request SSL certificate từ Let's Encrypt
6. ✅ Build và start containers
7. ✅ Health check

**Thời gian**: 5-10 phút (tùy tốc độ mạng)

### 3.4. Theo dõi quá trình

Script sẽ hiển thị từng bước:
```
================================================
DigitalOcean Droplet Deployment
================================================
Droplet IP: 165.227.xxx.xxx
Domain: yourdomain.com
...

Step 1: Testing SSH connection...
✅ SSH connection working

Step 2: Copying files to droplet...
✅ Files copied successfully

Step 3: Updating configuration...
✅ Configuration updated

Step 4: Running setup script on droplet...
This may take 5-10 minutes...
...
```

### 3.5. Nếu gặp lỗi

**Lỗi SSH:**
```powershell
# Thử connect thủ công
ssh root@YOUR_DROPLET_IP

# Nếu được, script sẽ chạy được
```

**Lỗi SSL Certificate:**
- DNS chưa propagate → Chờ 30 phút, chạy lại
- Email sai → Check email có đúng không

**Lỗi Docker:**
- Thử chạy lại script (tự động fix)

---

## ✅ BƯỚC 4: Kiểm tra (2 phút)

### 4.1. Kiểm tra từ trình duyệt

1. Mở trình duyệt
2. Vào: `https://yourdomain.com`
   - **Kết quả mong đợi**: Thấy Streamlit Dashboard
   - **Có khóa xanh** ở thanh địa chỉ (SSL working)

3. Vào: `https://yourdomain.com/docs`
   - **Kết quả mong đợi**: Thấy FastAPI Swagger UI

### 4.2. Kiểm tra từ command line

```powershell
# Test API
curl https://yourdomain.com/health

# Kết quả:
# {"status":"healthy"}

# Test Dashboard
curl https://yourdomain.com

# Kết quả: HTML của Streamlit
```

### 4.3. Kiểm tra trên server

```powershell
# SSH vào server
ssh root@YOUR_DROPLET_IP

# Kiểm tra containers
cd /opt/autoscaling-analysis/digitalocean
sudo docker-compose -f docker-compose.droplet.yml ps

# Kết quả mong đợi:
# NAME                  STATUS              PORTS
# autoscaling-api       Up 2 minutes        8000/tcp
# autoscaling-dashboard Up 2 minutes        8501/tcp
# nginx                 Up 2 minutes        0.0.0.0:80->80/tcp, 0.0.0.0:443->443/tcp
# certbot               Up 2 minutes

# Xem logs
sudo docker-compose -f docker-compose.droplet.yml logs -f

# Nhấn Ctrl+C để thoát logs
```

---

## 🎉 Hoàn thành!

### Bạn đã deploy thành công nếu:
- ✅ `https://yourdomain.com` → Dashboard hoạt động
- ✅ `https://yourdomain.com/docs` → API docs hoạt động
- ✅ Có khóa xanh SSL
- ✅ Không có cảnh báo security

### Thông tin truy cập:
- 🌐 **Dashboard**: https://yourdomain.com
- 🔌 **API Docs**: https://yourdomain.com/docs
- ❤️ **Health Check**: https://yourdomain.com/health
- 🖥️ **SSH**: `ssh root@YOUR_DROPLET_IP`

### Chi phí:
- 💰 **Droplet**: $6-12/tháng
- 🔐 **SSL**: $0 (Let's Encrypt miễn phí)
- 🌐 **Bandwidth**: 1-2 TB free
- **Tổng**: $6-12/tháng

---

## 🔧 Quản lý hàng ngày

### Xem logs
```powershell
ssh root@YOUR_DROPLET_IP
cd /opt/autoscaling-analysis/digitalocean

# Xem tất cả logs
sudo docker-compose -f docker-compose.droplet.yml logs -f

# Chỉ xem API logs
sudo docker-compose -f docker-compose.droplet.yml logs -f api

# Chỉ xem Dashboard logs
sudo docker-compose -f docker-compose.droplet.yml logs -f dashboard

# Xem 100 dòng cuối
sudo docker-compose -f docker-compose.droplet.yml logs --tail=100
```

### Restart services
```powershell
ssh root@YOUR_DROPLET_IP
cd /opt/autoscaling-analysis/digitalocean

# Restart all
sudo docker-compose -f docker-compose.droplet.yml restart

# Restart API only
sudo docker-compose -f docker-compose.droplet.yml restart api

# Restart Dashboard only
sudo docker-compose -f docker-compose.droplet.yml restart dashboard
```

### Update code
```powershell
# 1. Push code mới lên GitHub (từ máy local)
git add .
git commit -m "Update feature"
git push origin main

# 2. SSH vào server
ssh root@YOUR_DROPLET_IP
cd /opt/autoscaling-analysis

# 3. Pull code mới
git pull origin main

# 4. Rebuild và restart
cd digitalocean
sudo docker-compose -f docker-compose.droplet.yml up -d --build

# 5. Kiểm tra
sudo docker-compose -f docker-compose.droplet.yml ps
sudo docker-compose -f docker-compose.droplet.yml logs -f
```

### Stop services (tiết kiệm tài nguyên)
```powershell
ssh root@YOUR_DROPLET_IP
cd /opt/autoscaling-analysis/digitalocean

# Stop all
sudo docker-compose -f docker-compose.droplet.yml down

# Start lại
sudo docker-compose -f docker-compose.droplet.yml up -d
```

### Backup
```powershell
# Từ máy local, backup models
scp -r root@YOUR_DROPLET_IP:/opt/autoscaling-analysis/models ./backup/

# Backup toàn bộ app
ssh root@YOUR_DROPLET_IP "cd /opt && tar czf autoscaling-backup-$(date +%Y%m%d).tar.gz autoscaling-analysis"
scp root@YOUR_DROPLET_IP:/opt/autoscaling-backup-*.tar.gz ./
```

---

## 🐛 Troubleshooting

### Website không mở được

**1. Kiểm tra DNS**
```powershell
nslookup yourdomain.com

# Phải thấy IP của droplet
# Nếu không → Chờ DNS propagate hoặc check lại DNS config
```

**2. Kiểm tra services**
```powershell
ssh root@YOUR_DROPLET_IP
cd /opt/autoscaling-analysis/digitalocean
sudo docker-compose -f docker-compose.droplet.yml ps

# Tất cả phải "Up"
# Nếu "Exit" hoặc "Restarting" → Xem logs
sudo docker-compose -f docker-compose.droplet.yml logs
```

**3. Kiểm tra ports**
```powershell
ssh root@YOUR_DROPLET_IP

# Kiểm tra ports đang listen
sudo netstat -tulpn | grep -E ':(80|443|8000|8501)'

# Phải thấy:
# :80 (nginx)
# :443 (nginx)
# :8000 (api)
# :8501 (dashboard)
```

### SSL không hoạt động

**1. Kiểm tra certificate**
```powershell
ssh root@YOUR_DROPLET_IP
cd /opt/autoscaling-analysis/digitalocean

# List certificates
sudo ls -la certbot/conf/live/

# Phải thấy thư mục yourdomain.com
```

**2. Request lại certificate**
```powershell
ssh root@YOUR_DROPLET_IP
cd /opt/autoscaling-analysis/digitalocean

# Stop nginx
sudo docker-compose -f docker-compose.droplet.yml stop nginx

# Request certificate
sudo docker run --rm \
    -v $(pwd)/certbot/conf:/etc/letsencrypt \
    -v $(pwd)/certbot/www:/var/www/certbot \
    certbot/certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email your@email.com \
    --agree-tos \
    --no-eff-email \
    -d yourdomain.com \
    -d www.yourdomain.com

# Start lại
sudo docker-compose -f docker-compose.droplet.yml up -d
```

### API hoặc Dashboard không hoạt động

**1. Xem logs chi tiết**
```powershell
ssh root@YOUR_DROPLET_IP
cd /opt/autoscaling-analysis/digitalocean

# API logs
sudo docker-compose -f docker-compose.droplet.yml logs api | tail -50

# Dashboard logs
sudo docker-compose -f docker-compose.droplet.yml logs dashboard | tail -50
```

**2. Restart container**
```powershell
sudo docker-compose -f docker-compose.droplet.yml restart api
sudo docker-compose -f docker-compose.droplet.yml restart dashboard
```

**3. Rebuild từ đầu**
```powershell
sudo docker-compose -f docker-compose.droplet.yml down
sudo docker-compose -f docker-compose.droplet.yml up -d --build
```

### Out of memory

**Nâng cấp Droplet:**
1. Vào DigitalOcean Dashboard
2. Click vào Droplet
3. **Resize** → Chọn plan lớn hơn (2GB → 4GB)
4. **Resize Droplet**

**Hoặc thêm swap:**
```powershell
ssh root@YOUR_DROPLET_IP

# Tạo 2GB swap
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Permanent swap
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Kiểm tra
free -h
```

### Droplet bị hack hoặc hacked

**Bảo mật cơ bản:**
```powershell
ssh root@YOUR_DROPLET_IP

# 1. Setup firewall
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable

# 2. Disable password login (nếu dùng SSH key)
sudo nano /etc/ssh/sshd_config
# Tìm và thay đổi:
# PasswordAuthentication no
# PubkeyAuthentication yes
sudo systemctl restart sshd

# 3. Install fail2ban
sudo apt install fail2ban -y
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# 4. Automatic security updates
sudo apt install unattended-upgrades -y
sudo dpkg-reconfigure --priority=low unattended-upgrades
```

---

## 📞 Cần giúp đỡ?

### Các lệnh hữu ích

```powershell
# Kiểm tra disk space
ssh root@YOUR_DROPLET_IP "df -h"

# Kiểm tra memory
ssh root@YOUR_DROPLET_IP "free -h"

# Kiểm tra CPU
ssh root@YOUR_DROPLET_IP "top -n 1"

# Xem Docker images
ssh root@YOUR_DROPLET_IP "sudo docker images"

# Clean up Docker
ssh root@YOUR_DROPLET_IP "sudo docker system prune -a"

# Restart Docker daemon
ssh root@YOUR_DROPLET_IP "sudo systemctl restart docker"
```

### Logs quan trọng

```powershell
# Nginx logs
ssh root@YOUR_DROPLET_IP
sudo docker-compose -f docker-compose.droplet.yml logs nginx

# System logs
ssh root@YOUR_DROPLET_IP
sudo journalctl -xe

# Docker daemon logs
ssh root@YOUR_DROPLET_IP
sudo journalctl -u docker
```

---

## 🎓 Video hướng dẫn (nếu cần)

Nếu bạn muốn, tôi có thể tạo thêm:
- ✅ Video recording các bước
- ✅ Screenshots từng bước
- ✅ Troubleshooting guide chi tiết hơn

---

**Chúc bạn deploy thành công! 🚀**

Nếu gặp bất kỳ lỗi nào, paste lỗi đó cho tôi, tôi sẽ giúp bạn fix!
