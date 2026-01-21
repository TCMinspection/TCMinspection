# 🌐 华为云部署指南 - 中医AI智能望诊系统

## 📋 目录
- [部署概述](#部署概述)
- [前置要求](#前置要求)
- [华为云资源准备](#华为云资源准备)
- [部署步骤](#部署步骤)
- [配置说明](#配置说明)
- [监控和维护](#监控和维护)
- [故障排除](#故障排除)
- [成本估算](#成本估算)

## 🎯 部署概述

本指南详细说明如何将中医AI智能望诊系统部署到华为云平台。华为云提供完整的云计算服务，特别适合MindSpore生态的AI应用。

### 部署架构
```
用户 → 华为云CDN → 负载均衡 → 弹性云服务器ECS
                           ↓
                       容器引擎CCE
                           ↓
                       对象存储OBS
                           ↓
                       云数据库RDS
```

## 📚 前置要求

### 技术要求
- Ubuntu 20.04 LTS 操作系统知识
- 基础的Linux命令行操作能力
- Python 3.8+ 环境管理
- 域名管理（可选）

### 华为云账号要求
- 已注册华为云账号
- 完成实名认证
- 账户余额充足

## 🚀 华为云资源准备

### 1. 注册华为云账号
1. 访问 [华为云官网](https://www.huaweicloud.com)
2. 点击"注册"完成账号注册
3. 进行实名认证（个人或企业）
4. 充值账户（建议预存100元以上）

### 2. 创建访问密钥
1. 登录华为云控制台
2. 点击右上角头像 → "我的凭证"
3. 选择"访问密钥" → "新增访问密钥"
4. 下载CSV文件并妥善保存
5. 记录 Access Key ID 和 Secret Access Key

### 3. 查看项目ID
1. 在"我的凭证"页面
2. 查看"项目列表"
3. 记录目标项目的项目ID

### 4. 购买ECS云服务器

#### 推荐配置
| 配置项 | 推荐值 | 说明 |
|--------|--------|------|
| 计算型实例 | c6.xlarge.4 | 4核16GB，适合AI推理 |
| 操作系统 | Ubuntu 20.04 LTS | 稳定性好 |
| 系统盘 | 通用型SSD 100GB | 系统和基础软件 |
| 数据盘 | 通用型SSD 200GB | 模型文件和数据存储 |
| 网络 | 带宽10Mbps | 支持图片上传 |
| 安全组 | 开放80/443/22端口 | HTTP/HTTPS/SSH |

#### 购买步骤
1. 进入华为云控制台
2. 选择"弹性云服务器 ECS"
3. 点击"购买弹性云服务器"
4. 按推荐配置选择参数
5. 确认订单并支付

### 5. 配置安全组
1. 找到购买的ECS实例
2. 点击"安全组" → "配置规则"
3. 添加入方向规则：
   - 端口: 22, 协议: TCP, 源地址: 0.0.0.0/0 (SSH)
   - 端口: 80, 协议: TCP, 源地址: 0.0.0.0/0 (HTTP)
   - 端口: 443, 协议: TCP, 源地址: 0.0.0.0/0 (HTTPS)

## 🛠️ 部署步骤

### 第一步：连接到ECS服务器

```bash
# 使用SSH连接（替换为您的ECS公网IP）
ssh root@YOUR_ECS_PUBLIC_IP

# 创建部署用户
adduser tcmuser
usermod -aG sudo tcmuser
su - tcmuser
```

### 第二步：上传部署文件

将以下文件上传到服务器：
- `huaweicloud-deploy.sh` - 主部署脚本
- `huaweicloud-config.env` - 环境配置模板
- `nginx-huaweicloud.conf` - Nginx配置
- `huaweicloud-obs-integration.py` - OBS集成脚本

```bash
# 在本地执行，上传文件到服务器
scp deploy/* tcmuser@YOUR_ECS_PUBLIC_IP:/home/tcmuser/
```

### 第三步：执行部署脚本

```bash
# 在ECS服务器上执行
cd /home/tcmuser
chmod +x huaweicloud-deploy.sh

# 设置环境变量（可选）
export GIT_REPO="https://github.com/your-username/tcm-inspection.git"

# 执行部署脚本
./huaweicloud-deploy.sh
```

### 第四步：配置环境变量

编辑环境配置文件：
```bash
nano /home/tcmuser/tcm-inspection/.env
```

重要配置项：
```bash
# 华为云认证
HUAWEICLOUD_SDK_AK=your-access-key-here
HUAWEICLOUD_SDK_SK=your-secret-key-here
HUAWEICLOUD_SDK_PROJECT_ID=your-project-id-here

# OpenAI API
OPENAI_API_KEY=sk-or-v1-your-openai-api-key-here

# 域名（如果有的话）
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com
```

### 第五步：上传模型文件

将训练好的模型文件上传到服务器：
```bash
# 创建模型目录
mkdir -p /home/tcmuser/tcm-inspection/models

# 上传模型文件（在本地执行）
scp autoencoder_model_mindspore.ckpt tcmuser@YOUR_ECS_PUBLIC_IP:/home/tcmuser/tcm-inspection/models/
scp svm_pipeline.pkl tcmuser@YOUR_ECS_PUBLIC_IP:/home/tcmuser/tcm-inspection/models/
```

### 第六步：启动应用

```bash
# 重启应用
supervisorctl restart tcm-inspection

# 检查状态
supervisorctl status tcm-inspection

# 查看日志
tail -f /home/tcmuser/tcm-inspection/logs/app.log
```

## ⚙️ 配置说明

### Nginx配置
编辑Nginx配置文件：
```bash
sudo nano /etc/nginx/sites-available/tcm-inspection
```

替换 `your-domain.com` 为您的实际域名，然后重新加载Nginx：
```bash
sudo nginx -t
sudo systemctl reload nginx
```

### SSL证书配置

#### 使用Let's Encrypt（免费）
```bash
# 安装Certbot
sudo apt install certbot python3-certbot-nginx

# 获取SSL证书
sudo certbot --nginx -d your-domain.com

# 设置自动续期
sudo crontab -e
# 添加：0 12 * * * /usr/bin/certbot renew --quiet
```

### 华为云OBS配置（可选）

如果需要使用对象存储备份：

1. 创建OBS桶：
   - 进入华为云控制台 → "对象存储服务 OBS"
   - 创建桶：`tcm-inspection-bucket`

2. 配置OBS权限：
   - 在桶的"权限设置"中设置公共读取权限（如需要）

3. 测试OBS集成：
```bash
cd /home/tcmuser/tcm-inspection
python3 deploy/huaweicloud-obs-integration.py
```

## 📊 监控和维护

### 系统监控

#### 1. 应用状态监控
```bash
# 查看应用状态
supervisorctl status tcm-inspection

# 查看系统资源
htop
df -h
free -h
```

#### 2. 日志监控
```bash
# 应用日志
tail -f /home/tcmuser/tcm-inspection/logs/app.log

# Nginx访问日志
sudo tail -f /var/log/nginx/tcm_inspection_access.log

# Nginx错误日志
sudo tail -f /var/log/nginx/tcm_inspection_error.log
```

#### 3. 健康检查
```bash
# 本地健康检查
curl http://localhost/health

# 远程健康检查
curl http://YOUR_ECS_PUBLIC_IP/health
```

### 备份策略

#### 自动备份配置
系统已配置自动备份：
- 每5分钟执行健康检查
- 每日凌晨2点执行OBS备份
- 日志文件自动轮转（保留30天）

#### 手动备份
```bash
# 备份整个项目
tar -czf /backup/tcm-inspection-$(date +%Y%m%d).tar.gz /home/tcmuser/tcm-inspection

# 备份到OBS
cd /home/tcmuser/tcm-inspection
python3 deploy/huaweicloud-obs-integration.py
```

### 性能优化

#### 1. 系统优化
```bash
# 调整文件描述符限制
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# 优化内核参数
echo "net.core.somaxconn = 65536" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

#### 2. 应用优化
- 根据服务器规格调整Gunicorn worker数量
- 启用Nginx缓存
- 配置适当的超时时间

## 🔧 故障排除

### 常见问题

#### 1. 应用无法启动
```bash
# 检查应用状态
supervisorctl status tcm-inspection

# 查看详细日志
supervisorctl tail -f tcm-inspection

# 重启应用
supervisorctl restart tcm-inspection
```

#### 2. 502 Bad Gateway错误
```bash
# 检查后端服务是否运行
ps aux | grep gunicorn

# 检查端口是否被占用
netstat -tlnp | grep :8000

# 重启Nginx
sudo systemctl restart nginx
```

#### 3. 模型加载失败
- 检查模型文件是否存在：`ls -la /home/tcmuser/tcm-inspection/models/`
- 检查文件权限：`chmod 644 /home/tcmuser/tcm-inspection/models/*`
- 查看应用日志中的错误信息

#### 4. API调用失败
- 检查OpenAI API密钥配置
- 验证网络连接：`curl -I https://openrouter.ai/api/v1`
- 查看应用日志中的API错误

#### 5. 内存不足
```bash
# 检查内存使用
free -h
ps aux --sort=-%mem | head

# 重启应用释放内存
supervisorctl restart tcm-inspection
```

### 日志分析

#### 应用日志位置
- 应用日志：`/home/tcmuser/tcm-inspection/logs/app.log`
- Gunicorn日志：`/home/tcmuser/tcm-inspection/logs/gunicorn_*.log`
- Supervisor日志：`/home/tcmuser/tcm-inspection/logs/supervisor.log`

#### 关键日志查询
```bash
# 查找错误日志
grep "ERROR" /home/tcmuser/tcm-inspection/logs/app.log

# 查找API调用日志
grep "api" /home/tcmuser/tcm-inspection/logs/app.log

# 实时监控日志
tail -f /home/tcmuser/tcm-inspection/logs/app.log | grep -E "(ERROR|WARN)"
```

## 💰 成本估算

### 华为云服务费用（月度）

| 服务 | 配置 | 预估费用 |
|------|------|----------|
| ECS云服务器 | 4核16GB，300GB SSD | 200-400元 |
| 弹性公网IP | 10Mbps带宽 | 50-100元 |
| 对象存储OBS | 100GB存储 | 10-20元 |
| 域名（可选） | .com域名 | 50-100元/年 |
| **月度总计** | | **260-520元** |

### 年度费用估算
- 基础服务：约3,120-6,240元/年
- 域名费用：约50-100元/年
- **年度总计**：约3,170-6,340元/年

### 成本优化建议

1. **按需付费**：初期可以使用按需付费模式
2. **包年包月**：稳定运行后购买包年包月更优惠
3. **资源监控**：定期检查资源使用情况，避免浪费
4. **备份策略**：合理使用OBS生命周期策略降低存储成本

## 🆘 技术支持

### 华为云技术支持
- 官方文档：https://support.huaweicloud.com/
- 工单系统：华为云控制台 → "工单" → "提交工单"
- 技术支持电话：950808

### 常用链接
- 华为云控制台：https://console.huaweicloud.com/
- ECS文档：https://support.huaweicloud.com/ecs/
- OBS文档：https://support.huaweicloud.com/obs/

### 部署验证清单

部署完成后，请验证以下功能：

- [ ] 网站可以通过公网IP访问
- [ ] 图片上传功能正常
- [ ] AI模型推理正常
- [ ] 智能对话功能正常
- [ ] 检测报告生成正常
- [ ] SSL证书正常工作（如果配置了域名）
- [ ] 监控脚本正常运行
- [ ] 备份功能正常
- [ ] 日志记录正常

---

**🎉 恭喜！您的中医AI智能望诊系统已成功部署到华为云！**

如有任何问题，请参考故障排除部分或联系技术支持。