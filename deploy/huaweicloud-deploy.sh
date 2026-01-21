#!/bin/bash
# 华为云中医AI智能望诊系统部署脚本
# 作者：AI Assistant
# 版本：1.0

set -e

# 配置参数
PROJECT_NAME="tcm-inspection"
DEPLOY_USER="tcmuser"
DEPLOY_PATH="/home/${DEPLOY_USER}/${PROJECT_NAME}"
PYTHON_VERSION="3.8"

echo "🏥 华为云中医AI智能望诊系统部署脚本"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否为root用户
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "请不要使用root用户运行此脚本"
        exit 1
    fi
}

# 系统更新
update_system() {
    log_info "更新系统软件包..."
    sudo apt update && sudo apt upgrade -y
}

# 安装基础依赖
install_dependencies() {
    log_info "安装基础依赖..."
    sudo apt install -y python3 python3-pip python3-venv nginx git \
        curl wget unzip htop iotop build-essential \
        libopencv-dev python3-opencv

    # 安装华为云SDK
    pip3 install huaweicloud-sdk-core huaweicloud-sdk-ecs huaweicloud-sdk-obs
}

# 创建项目目录
create_project_dir() {
    log_info "创建项目目录..."
    mkdir -p ${DEPLOY_PATH}
    mkdir -p ${DEPLOY_PATH}/{uploads,temp,reports,logs,models}
    mkdir -p ${DEPLOY_PATH}/conversation_memory
}

# 部署项目代码
deploy_code() {
    log_info "部署项目代码..."

    # 如果是git仓库
    if [ -n "$GIT_REPO" ]; then
        log_info "从Git仓库拉取代码..."
        git clone $GIT_REPO ${DEPLOY_PATH}
    else
        log_info "请手动上传项目代码到 ${DEPLOY_PATH}"
        read -p "代码上传完成后按回车继续..."
    fi

    # 设置权限
    chown -R ${DEPLOY_USER}:${DEPLOY_USER} ${DEPLOY_PATH}
    chmod -R 755 ${DEPLOY_PATH}
}

# 配置Python环境
setup_python_env() {
    log_info "配置Python虚拟环境..."

    cd ${DEPLOY_PATH}
    python3 -m venv venv
    source venv/bin/activate

    # 升级pip
    pip install --upgrade pip

    # 安装项目依赖
    if [ -f "requirements.txt" ]; then
        log_info "安装Python依赖包..."
        pip install -r requirements.txt
    else
        log_warn "未找到requirements.txt文件，请手动创建"
    fi

    # 安装生产环境依赖
    pip install gunicorn supervisor
}

# 配置环境变量
setup_env() {
    log_info "配置环境变量..."

    cat > ${DEPLOY_PATH}/.env << EOF
# 华为云部署环境配置
FLASK_ENV=production
SECRET_KEY=$(openssl rand -hex 32)
OPENAI_API_KEY=sk-or-v1-your-openai-api-key-here

# 华为云配置
HUAWEICLOUD_SDK_AK=your-huawei-ak
HUAWEICLOUD_SDK_SK=your-huawei-sk
HUAWEICLOUD_SDK_PROJECT_ID=your-project-id

# 应用配置
MAX_CONTENT_LENGTH=16777216
UPLOAD_FOLDER=${DEPLOY_PATH}/uploads
TEMP_FOLDER=${DEPLOY_PATH}/temp
REPORTS_FOLDER=${DEPLOY_PATH}/reports
MODELS_FOLDER=${DEPLOY_PATH}/models

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=${DEPLOY_PATH}/logs/app.log

# 华为云OBS配置（可选）
OBS_BUCKET_NAME=tcm-inspection-bucket
OBS_ENDPOINT=https://obs.cn-north-4.myhuaweicloud.com
EOF

    chmod 600 ${DEPLOY_PATH}/.env
}

# 配置Gunicorn
setup_gunicorn() {
    log_info "配置Gunicorn..."

    cat > ${DEPLOY_PATH}/gunicorn.conf.py << EOF
# Gunicorn配置文件 - 华为云优化版
import multiprocessing

# 服务器套接字
bind = "127.0.0.1:8000"
backlog = 2048

# 工作进程
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
preload_app = True
timeout = 60
keepalive = 2

# 日志
accesslog = "${DEPLOY_PATH}/logs/gunicorn_access.log"
errorlog = "${DEPLOY_PATH}/logs/gunicorn_error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 进程命名
proc_name = 'tcm-inspection'

# 安全
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# 华为云优化
worker_tmp_dir = /dev/shm
EOF
}

# 配置Supervisor
setup_supervisor() {
    log_info "配置Supervisor进程管理..."

    cat > /tmp/${PROJECT_NAME}.conf << EOF
[program:${PROJECT_NAME}]
command=${DEPLOY_PATH}/venv/bin/gunicorn --config gunicorn.conf.py app:app
directory=${DEPLOY_PATH}
user=${DEPLOY_USER}
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=${DEPLOY_PATH}/logs/supervisor.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=5
environment=PATH="${DEPLOY_PATH}/venv/bin"
EOF

    sudo mv /tmp/${PROJECT_NAME}.conf /etc/supervisor/conf.d/
    sudo supervisorctl reread
    sudo supervisorctl update
}

# 配置Nginx
setup_nginx() {
    log_info "配置Nginx反向代理..."

    cat > /tmp/${PROJECT_NAME} << EOF
server {
    listen 80;
    server_name _;

    # 安全头部
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # 文件上传大小限制
    client_max_body_size 16M;

    # 静态文件缓存
    location ~* \.(css|js|ico|png|jpg|jpeg|gif|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        add_header X-Static-File yes;
    }

    # API请求代理
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # WebSocket支持（如果需要）
    location /ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # 主应用代理
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # 健康检查
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF

    sudo mv /tmp/${PROJECT_NAME} /etc/nginx/sites-available/
    sudo rm -f /etc/nginx/sites-enabled/default
    sudo ln -s /etc/nginx/sites-available/${PROJECT_NAME} /etc/nginx/sites-enabled/
    sudo nginx -t
    sudo systemctl restart nginx
}

# 配置防火墙
setup_firewall() {
    log_info "配置防火墙..."
    sudo ufw allow ssh
    sudo ufw allow 'Nginx Full'
    sudo ufw --force enable
}

# 配置日志轮转
setup_logrotate() {
    log_info "配置日志轮转..."

    cat > /tmp/${PROJECT_NAME} << EOF
${DEPLOY_PATH}/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 ${DEPLOY_USER} ${DEPLOY_USER}
    postrotate
        supervisorctl restart ${PROJECT_NAME}
    endscript
}
EOF

    sudo mv /tmp/${PROJECT_NAME} /etc/logrotate.d/
}

# 配置监控脚本
setup_monitoring() {
    log_info "配置监控脚本..."

    cat > ${DEPLOY_PATH}/scripts/health_check.sh << 'EOF'
#!/bin/bash
# 健康检查脚本

APP_URL="http://127.0.0.1:8000/api/health"
LOG_FILE="/home/tcmuser/tcm-inspection/logs/health_check.log"

response=$(curl -s -o /dev/null -w "%{http_code}" $APP_URL)

if [ $response -eq 200 ]; then
    echo "$(date): Application is healthy" >> $LOG_FILE
else
    echo "$(date): Application is unhealthy (HTTP $response)" >> $LOG_FILE
    # 重启应用
    supervisorctl restart tcm-inspection
fi
EOF

    chmod +x ${DEPLOY_PATH}/scripts/health_check.sh

    # 添加到crontab
    (crontab -l 2>/dev/null; echo "*/5 * * * * ${DEPLOY_PATH}/scripts/health_check.sh") | crontab -
}

# 华为云OBS备份配置
setup_obs_backup() {
    log_info "配置华为云OBS备份..."

    cat > ${DEPLOY_PATH}/scripts/obs_backup.sh << EOF
#!/bin/bash
# 华为云OBS备份脚本

BACKUP_DIR="/tmp/tcm_inspection_backup_\$(date +%Y%m%d_%H%M%S)"
OBS_BUCKET="\$OBS_BUCKET_NAME"

# 创建备份
mkdir -p \$BACKUP_DIR
cp -r ${DEPLOY_PATH}/reports \$BACKUP_DIR/
cp -r ${DEPLOY_PATH}/conversation_memory \$BACKUP_DIR/
cp -r ${DEPLOY_PATH}/models \$BACKUP_DIR/

# 上传到OBS
if command -v obsutil &> /dev/null; then
    obsutil cp -r \$BACKUP_DIR obs://\$OBS_BUCKET/backups/
    rm -rf \$BACKUP_DIR
    echo "\$(date): 备份上传成功" >> ${DEPLOY_PATH}/logs/backup.log
else
    echo "\$(date): obsutil未安装，备份失败" >> ${DEPLOY_PATH}/logs/backup.log
fi
EOF

    chmod +x ${DEPLOY_PATH}/scripts/obs_backup.sh

    # 添加每日备份任务
    (crontab -l 2>/dev/null; echo "0 2 * * * ${DEPLOY_PATH}/scripts/obs_backup.sh") | crontab -
}

# 主函数
main() {
    log_info "开始部署中医AI智能望诊系统到华为云..."

    check_root
    update_system
    install_dependencies
    create_project_dir
    deploy_code
    setup_python_env
    setup_env
    setup_gunicorn
    setup_supervisor
    setup_nginx
    setup_firewall
    setup_logrotate
    setup_monitoring
    setup_obs_backup

    log_info "部署完成！"
    echo ""
    echo "🎉 部署成功！"
    echo "=================================="
    echo "应用地址: http://$(curl -s ifconfig.me)"
    echo "健康检查: http://$(curl -s ifconfig.me)/health"
    echo "项目目录: ${DEPLOY_PATH}"
    echo "日志目录: ${DEPLOY_PATH}/logs"
    echo ""
    echo "下一步操作："
    echo "1. 配置环境变量文件: ${DEPLOY_PATH}/.env"
    echo "2. 上传模型文件到: ${DEPLOY_PATH}/models/"
    echo "3. 重启应用: supervisorctl restart ${PROJECT_NAME}"
    echo "4. 配置域名和SSL证书"
    echo ""
    echo "监控命令："
    echo "- 查看应用状态: supervisorctl status ${PROJECT_NAME}"
    echo "- 查看应用日志: tail -f ${DEPLOY_PATH}/logs/app.log"
    echo "- 查看Nginx日志: sudo tail -f /var/log/nginx/access.log"
}

# 执行主函数
main "$@"