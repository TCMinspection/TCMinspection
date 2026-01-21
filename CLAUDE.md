# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码仓库中工作提供指导。

**📝 文档规范**：所有CLAUDE.md文档内容请使用中文编写。

## 项目概述

中医AI智能望诊系统是一个基于人工智能的中医诊断辅助系统，专注于通过面部面色分析来提供健康状况评估和个性化调理建议。该系统结合传统中医理论与现代机器学习技术，实现了自动化的面色识别、智能诊断报告生成以及交互式健康咨询功能。

## 开发命令

### 快速启动命令
```bash
# 带环境检查启动系统（推荐）
python run.py

# 直接启动Flask Web应用
python app.py

# 测试系统并修复导入问题后启动
python test_new_import.py && python app.py
```

### 测试和故障排除
```bash
# 测试模块导入并诊断问题
python test_new_import.py
python module_imports.py

# 修复常见导入问题
python fix_imports.py

# 单独测试SVM功能
python test_svm_fix.py
```

### 模型训练和准备
```bash
# 训练自编码器用于特征压缩
cd code.part1
python Autoencoder_mindspore.py

# 训练SVM分类器
cd code.part2
python SVM未修改版.py

# 使用训练好的自编码器提取特征
cd code.part1
python 提取特征_mindspore.py
```

## 系统架构总览

整个系统采用模块化设计，分为四个核心部分，每个部分承担不同的功能职责，形成完整的面色诊断流程。

### 核心技术栈

**深度学习框架**：
- MindSpore（主要）- 华为深度学习框架
- PyTorch（备用）- 替代框架支持

**机器学习**：
- scikit-learn - SVM分类和预处理管道
- OpenCV - 图像处理和面部检测
- dlib - 面部关键点检测

**AI和LLM**：
- LangChain - AI编排框架
- OpenAI GPT via OpenRouter API - 报告生成和对话
- FAISS - 向量相似性搜索用于RAG

**Web框架**：
- Flask - REST API服务器
- HTML5 + Tailwind CSS - 现代化前端界面
- AOS (Animate On Scroll) - 滚动动画库
- RemixIcon - 图标库
- JavaScript ES6+ - 前端交互逻辑

## 核心模块架构

### 1. 特征提取与降维模块 (code.part1)

**功能定位**：负责从面部图像中提取高质量的特征表示，并通过自编码器进行特征降维

**技术架构**：
- **双框架支持**：同时支持PyTorch和MindSpore两种深度学习框架
- **CNN特征提取器**：使用预训练的卷积神经网络提取4096维高维特征
- **监督式自编码器**：结合分类损失的降维模型，将特征降维至128维
- **特征维度**：原始特征4096维 → 编码特征128维 → 分类特征5类

**核心组件**：
- `Autoencoder.py`：PyTorch版本的监督自编码器实现
- `Autoencoder_mindspore.py`：MindSpore版本的监督自编码器实现
- `提取特征_mindspore.py`：专门用于提取Autoencoder训练特征的工具脚本
- `分割面部区域*.py`：面部区域分割和预处理模块

**技术特点**：
- 采用监督学习策略，结合重构损失和分类损失
- 支持端到端的特征提取和降维流程
- 具备模型权重保存和特征提取功能

### 2. 分类识别模块 (code.part2)

**功能定位**：基于提取的特征进行中医面色类型分类识别

**技术架构**：
- **SVM分类器**：使用支持向量机进行最终的分类决策
- **特征融合**：结合CNN特征（1152维）和手工特征（252维）
- **标准化处理**：针对不同特征类型分别进行标准化
- **Pipeline设计**：使用sklearn的Pipeline实现特征处理和分类一体化

**实现细节**：
- `SVM未修改版.py`：完整的SVM分类管道实现
- 采用线性核函数，C=10.0的配置
- 支持模型持久化，保存为`svm_pipeline.pkl`
- 分类准确率报告和评估功能

**分类体系**：识别5种中医面色类型
- 0: 红色 - 阴虚火旺型
- 1: 黑色 - 肾虚寒盛型
- 2: 白色 - 气血两虚型
- 3: 黄色 - 脾虚湿盛型
- 4: 青色 - 寒凝血瘀型

### 3. 报告生成与问答模块 (code.part4)

**功能定位**：基于识别结果生成详细的中医诊断报告，并提供智能问答服务

**技术架构**：
- **大语言模型集成**：基于OpenRouter API使用GPT模型
- **检索增强生成(RAG)**：集成中医知识库，提供专业知识支撑
- **多模态交互**：支持文本报告生成和对话式咨询
- **记忆管理**：实现会话级别的对话记忆功能

**核心组件**：
- `agent.py`：基础TCM分析代理，专注于报告生成
- `enhanced_agent.py`：增强对话代理，支持多轮对话和记忆
- `reports/`：存储生成的诊断报告
- `conversation_memory/`：管理对话历史和记忆

**功能特性**：
- **智能报告生成**：根据面色类型和患者信息生成个性化诊断报告
- **多轮对话支持**：支持连续的中医咨询和健康问答
- **意图识别**：自动识别用户咨询意图（用药、饮食、调理时间、生活方式等）
- **安全机制**：内置医疗安全审核，确保建议的专业性

**API集成**：
- 使用OpenRouter作为大模型API提供商
- 支持GPT系列模型的灵活切换
- 集成FAISS向量数据库实现知识检索

### 关键组件

**TCMAnalyzer类 (app.py:111-355)**
- 整合所有ML模型和AI组件的主要编排器
- 处理面部检测、特征提取、分类和报告生成
- 使用dlib进行面部关键点检测和区域提取

**5种中医面色类型**：
- 0: 红色 - 阴虚火旺（阴虚有热）
- 1: 黑色 - 肾虚寒盛（肾阳不足）
- 2: 白色 - 气血两虚（气血不足）
- 3: 黄色 - 脾虚湿盛（脾虚湿重）
- 4: 青色 - 寒凝血瘀（寒凝血瘀）

### 数据处理管道

```
输入图像 → 面部检测 → 区域提取（9个区域）→
VGG16特征提取（4096维）→ 自编码器压缩（128维）→
手工特征（252维）→ 特征融合 → SVM分类 →
中医分析 → LLM报告生成 → AI对话
```

### 模型文件和依赖

**必需模型文件**：
- `autoencoder_model_mindspore.ckpt` - 训练好的自编码器权重
- `svm_pipeline.pkl` - 完整的SVM分类管道
- `shape_predictor_68_face_landmarks.dat` - dlib面部关键点检测器

**外部依赖**：
- 从以下地址下载dlib预测器：http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
- 解压后放置在 `D:/dlib-predictor/` 或项目根目录

### API配置

**OpenRouter API设置**：
```python
# 在agent.py和enhanced_agent.py中设置
os.environ["OPENAI_API_KEY"] = "sk-or-v1-your-api-key-here"

# API端点配置
OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4"
```

### 关键集成模式

**面部分析管道 (app.py:175-316)**：
```python
analyzer = get_analyzer()
result = analyzer.analyze_face_image(image_path)
# 返回：complexion_type, confidence, complexion_info
```

**中医报告生成 (agent.py:90-156)**：
```python
agent = TCMComplexionAnalysisAgent()
report = agent.quick_analysis(
    complexion_type=3,  # 黄色 - 脾虚湿盛
    confidence=0.87,
    patient_info={"name": "张三", "age": 35}
)
```

**AI对话系统 (enhanced_agent.py:1004)**：
```python
response = quick_chat_diagnosis(
    complexion_type=0,  # 红色 - 阴虚火旺
    user_question="饮食上需要注意什么？",
    patient_info={"name": "李四", "age": 30}
)
```

### 重要文件位置

- **生成的报告**：`./reports/` 目录
- **临时上传**：`./temp/` 和 `./uploads/` 目录
- **对话记忆**：`./conversation_memory/` 目录
- **HTML模板**：
  - `./templates/intro.html` - 中医知识介绍页面（首页）
  - `./templates/index.html` - AI智能望诊系统页面

### 4. Web应用模块 (code.part3) - 已废弃

**状态说明**：该模块为历史版本，当前已不再维护使用

**原始功能**：提供Web界面供用户上传面部图像进行诊断

**文件结构**：
- `app原版.py`：原始的Flask Web应用
- `index.html`：用户上传界面
- `result.html`：诊断结果展示页面

## 辅助模块架构

### 图像数据增强 (GAN模块)

**功能定位**：通过生成对抗网络进行训练数据扩增

**技术实现**：
- 使用GAN技术生成额外的面部图像样本
- 提高模型训练的泛化能力
- 解决中医面色数据集稀缺问题

### 工具和预处理

**Tools目录**：包含项目中使用的各种实用工具和预处理脚本
**dlib_predictor**：面部关键点检测和预处理工具
**imagesource相关**：管理训练和测试图像数据

## 数据流架构

### 训练数据流
```
原始图像 → 面部分割 → CNN特征提取 → 特征降维 → SVM训练 → 模型保存
```

### 推理数据流
```
用户上传 → 面部预处理 → 特征提取 → 分类识别 → 报告生成 → 问答交互
```

### 特征数据管理
- `feature*.npy`：各版本的特征文件
- `label*.npy`：对应的标签文件
- `encoder*.npy`：编码器提取的特征
- 支持PyTorch和MindSpore两个版本的特征格式

## 模型权重管理

**Autoencoder权重**：
- `autoencoder_model.pth`：PyTorch版本权重
- `autoencoder_model_mindspore.ckpt`：MindSpore版本权重

**SVM模型**：
- `svm_pipeline.pkl`：完整的SVM分类管道

## 技术栈总结

**深度学习框架**：PyTorch + MindSpore双框架支持
**机器学习**：scikit-learn (SVM, Pipeline, 预处理)
**大语言模型**：OpenAI GPT系列 (通过OpenRouter)
**AI编排**：LangChain框架
**向量检索**：FAISS
**Web框架**：Flask (历史版本)
**数据处理**：NumPy, OpenCV, dlib

## 部署架构特点

1. **国产化适配**：支持华为MindSpore框架，适配国产计算平台
2. **模块化设计**：各部分功能解耦，支持独立开发和部署
3. **双框架支持**：同时支持PyTorch和MindSpore，提供技术选择灵活性
4. **知识增强**：集成RAG系统，提供专业中医知识支撑
5. **安全机制**：内置医疗建议审核，确保使用安全

### 开发注意事项

1. **双框架支持**：系统同时支持MindSpore和PyTorch，但MindSpore是主要框架
2. **导入管理**：使用自定义模块导入系统（`module_imports.py`）处理跨模块依赖
3. **错误处理**：全面的错误处理和详细日志记录
4. **医疗安全**：内置安全检查，特定情况需要医生审核
5. **中文支持**：所有报告和回复都使用中文中医术语

### 环境要求

- Python 3.8+
- 8GB+ RAM（推荐16GB）
- 支持Windows/Linux/macOS
- 可选CUDA支持GPU加速

### 常见问题故障排除

**导入错误**：
```bash
python module_imports.py  # 诊断导入问题
python fix_imports.py     # 自动修复常见问题
```

**模型加载问题**：
- 检查`app.py:141-152`中的模型文件路径
- 验证dlib预测器是否正确放置
- 确保MindSpore正确安装

**API问题**：
- 验证OpenRouter API密钥配置
- 检查网络连接
- 监控API使用量和配额

### Web应用访问

系统提供完整的Web界面，采用现代化前后端分离架构：

#### 页面路由结构
- **中医知识介绍页面**：http://localhost:5000/ (首页)
- **AI智能望诊系统**：http://localhost:5000/analyze (检测页面)
- **API测试**：http://localhost:5000/test_api.html
- **健康检查**：http://localhost:5000/api/health

#### 中医知识介绍页面 (`templates/intro.html`)
**功能特色**：
- 🎨 **现代化设计**：采用渐变背景、毛玻璃效果和响应式布局
- 🎬 **丰富动效**：AOS滚动动画、数字递增动画、粒子背景效果
- 📱 **响应式适配**：完美支持桌面端和移动端设备
- 🧭 **平滑导航**：页面内锚点导航和平滑滚动效果

**页面结构**：
1. **英雄区域**：震撼标题展示中医望诊智慧与AI融合
2. **关于望诊**：详细解释中医望诊的核心理念和重要性
3. **五色主病**：图文并茂介绍五种面色类型及其对应病症
4. **历史传承**：时间线展示中医望诊从春秋战国到现代的发展历程
5. **科技融合**：展示AI技术如何与传统中医智慧结合
6. **开始体验**：引导用户进入AI智能检测系统

**技术特点**：
- 采用Tailwind CSS框架实现现代化UI设计
- 集成AOS动画库提供流畅的滚动动画效果
- 使用RemixIcon图标库增强视觉效果
- 支持移动端汉堡菜单和桌面端完整导航栏

#### AI智能望诊系统 (`templates/index.html`)
**核心功能**：
- 🖼️ **图像分析**：支持拖拽上传和点击上传面部照片
- 🤖 **智能咨询**：基于面色类型的AI中医健康问答
- 📋 **检测报告**：生成详细的中医健康评估报告
- 🔄 **双向导航**：可在介绍页面和检测系统间自由切换

**用户界面**：
- 标签页式设计，功能分区清晰
- 实时加载状态和错误提示
- 支持患者信息录入和管理
- 提供系统健康状态监控

#### 用户体验流程
1. **教育引导**：用户首先访问中医知识介绍页面，了解望诊文化背景
2. **无缝跳转**：通过"开始体验"按钮进入AI检测系统
3. **智能分析**：上传面部照片进行AI面色分析
4. **深度咨询**：基于分析结果进行个性化健康问答
5. **报告生成**：获得专业的中医健康评估报告

该架构设计充分考虑了中医诊断的特殊性，结合传统医学理论与现代AI技术，通过精心设计的用户界面和交互流程，为中医智能化提供了完整的技术解决方案和优秀的用户体验。

---

## 🌐 华为云部署方案

### 部署架构设计

**推荐方案：华为云ECS + OBS + Nginx + Supervisor**
- **计算服务**：华为云ECS弹性云服务器 (推荐c6.xlarge.4: 4核16GB)
- **前端服务**：Nginx (SSL终止 + 反向代理 + 静态文件服务)
- **应用服务**：Gunicorn + Flask + Supervisor进程管理
- **存储服务**：华为云OBS对象存储 (模型文件备份 + 用户数据)
- **监控服务**：华为云云监控 + 自定义健康检查
- **安全服务**：华为云安全组 + SSL证书 + WAF防护

### 系统架构图
```
用户 → 华为云CDN → 负载均衡 → 弹性云服务器ECS
                           ↓
                       容器引擎CCE
                           ↓
                       对象存储OBS
                           ↓
                       云数据库RDS
```

### 华为云资源准备清单

#### 1. 华为云账号配置
- 完成实名认证的华为云账号
- 账户余额充足（建议预存200元以上）
- 创建访问密钥 (Access Key ID + Secret Access Key)
- 记录项目ID (Project ID)

#### 2. 推荐资源配置
| 配置项 | 推荐值 | 说明 |
|--------|--------|------|
| ECS实例 | c6.xlarge.4 | 4核16GB，适合AI推理 |
| 操作系统 | Ubuntu 20.04 LTS | 稳定性好 |
| 系统盘 | 通用型SSD 100GB | 系统和基础软件 |
| 数据盘 | 通用型SSD 200GB | 模型文件和数据存储 |
| 网络 | 带宽10Mbps | 支持图片上传 |
| 安全组 | 开放80/443/22端口 | HTTP/HTTPS/SSH |

#### 3. 对象存储OBS配置
- 创建OBS桶：`tcm-inspection-bucket`
- 配置生命周期策略：自动清理旧备份
- 设置存储类别：标准存储

### 阶段一：华为云环境准备 (1天)

#### 1.1 创建ECS服务器
1. 进入华为云控制台 → 弹性云服务器ECS
2. 点击"购买弹性云服务器"
3. 按推荐配置选择参数
4. 确认订单并支付

#### 1.2 配置安全组
1. 找到购买的ECS实例
2. 点击"安全组" → "配置规则"
3. 添加入方向规则：
   - 端口: 22, 协议: TCP, 源地址: 0.0.0.0/0 (SSH)
   - 端口: 80, 协议: TCP, 源地址: 0.0.0.0/0 (HTTP)
   - 端口: 443, 协议: TCP, 源地址: 0.0.0.0/0 (HTTPS)

#### 1.3 连接服务器配置
```bash
# SSH连接到ECS服务器
ssh root@YOUR_ECS_PUBLIC_IP

# 创建部署用户
adduser tcmuser
usermod -aG sudo tcmuser
su - tcmuser
```

### 阶段二：应用部署配置 (1-2天)

#### 2.1 环境配置文件
已创建配置文件：
- `deploy/huaweicloud-config.env` - 华为云环境配置模板
- `deploy/huaweicloud-deploy.sh` - 华为云部署脚本
- `deploy/nginx-huaweicloud.conf` - 华为云优化Nginx配置

#### 2.2 关键环境变量
```bash
# 华为云认证
HUAWEICLOUD_SDK_AK=your-access-key-here
HUAWEICLOUD_SDK_SK=your-secret-key-here
HUAWEICLOUD_SDK_PROJECT_ID=your-project-id-here

# OpenAI API配置
OPENAI_API_KEY=sk-or-v1-your-openai-api-key-here
OPENAI_BASE_URL=https://openrouter.ai/api/v1

# 应用配置
FLASK_ENV=production
SECRET_KEY=your-secure-secret-key
DEBUG=False
```

#### 2.3 Supervisor进程管理
系统已配置Supervisor管理Flask应用：
- 自动重启机制
- 日志轮转
- 健康检查
- 性能监控

### 阶段三：AI模型部署 (1天)

#### 3.1 模型组件结构
系统包含以下AI模型组件：
- `autoencoder_model_mindspore.ckpt` - MindSpore自编码器（特征压缩）
- `svm_pipeline.pkl` - scikit-learn分类管道（最终分类）
- `shape_predictor_68_face_landmarks.dat` - dlib面部检测器（预处理）

#### 3.2 模型部署脚本
使用专用脚本 `deploy/model-deployment.sh`：
```bash
# 在本地执行模型部署
bash deploy/model-deployment.sh D:\TCMinspection

# 脚本功能：
# 1. 创建服务器模型目录结构
# 2. 上传所有必需模型文件
# 3. 配置dlib预测器
# 4. 验证模型完整性
# 5. 创建模型配置文件
# 6. 优化模型加载路径
# 7. 创建测试脚本
```

#### 3.3 模型文件目录结构
```
/home/tcmuser/tcm-inspection/models/
├── autoencoder_model_mindspore.ckpt    # MindSpore自编码器
├── svm_pipeline.pkl                    # SVM分类管道
├── dlib_predictor/
│   └── shape_predictor_68_face_landmarks.dat  # dlib面部检测器
├── model_config.json                   # 模型配置文件
└── test_models.py                      # 模型测试脚本
```

### 阶段四：OBS集成配置 (0.5天)

#### 4.1 OBS集成脚本
使用 `deploy/huaweicloud-obs-integration.py`：
```python
# 主要功能：
- 自动模型文件备份
- 用户数据备份
- 定时备份策略
- 生命周期管理
- 备份验证和恢复
```

#### 4.2 备份策略
- 模型文件：每日备份到OBS
- 用户数据：每小时增量备份
- 报告文件：每日完整备份
- 日志文件：每周备份，保留30天

### 阶段五：SSL证书和安全配置 (0.5天)

#### 5.1 SSL证书配置
```bash
# 安装Certbot
sudo apt install certbot python3-certbot-nginx

# 获取SSL证书
sudo certbot --nginx -d your-domain.com

# 设置自动续期
sudo crontab -e
# 添加：0 12 * * * /usr/bin/certbot renew --quiet
```

#### 5.2 安全增强配置
- Nginx安全头部配置
- 防火墙规则优化
- API访问限流
- 文件上传安全检查

### 华为云成本估算

#### 月度费用（人民币）
| 服务 | 配置 | 预估费用 |
|------|------|----------|
| ECS云服务器 | 4核16GB，300GB SSD | 200-400元 |
| 弹性公网IP | 10Mbps带宽 | 50-100元 |
| 对象存储OBS | 100GB存储 | 10-20元 |
| 域名（可选） | .com域名 | 50-100元/年 |
| **月度总计** | | **260-520元** |

#### 年度费用估算
- 基础服务：约3,120-6,240元/年
- 域名费用：约50-100元/年
- **年度总计**：约3,170-6,340元/年

### 部署时间表

- **第1天**：华为云账号配置和ECS创建
- **第2-3天**：应用部署和环境配置
- **第4天**：AI模型部署和验证
- **第4.5天**：OBS集成配置
- **第5天**：SSL证书和安全配置
- **第6天**：系统测试和性能优化

### 监控和运维

#### 1. 系统监控
```bash
# 查看应用状态
supervisorctl status tcm-inspection

# 查看系统资源
htop
df -h
free -h

# 健康检查
curl http://localhost/health
```

#### 2. 日志管理
- 应用日志：`/home/tcmuser/tcm-inspection/logs/app.log`
- Nginx访问日志：`/var/log/nginx/tcm_inspection_access.log`
- Nginx错误日志：`/var/log/nginx/tcm_inspection_error.log`
- Supervisor日志：`/home/tcmuser/tcm-inspection/logs/supervisor.log`

#### 3. 备份管理
```bash
# 手动备份模型到OBS
cd /home/tcmuser/tcm-inspection
python3 deploy/huaweicloud-obs-integration.py

# 验证备份状态
python3 -c "
from deploy.huaweicloud_obs_integration import HuaweiCloudOBSManager
manager = HuaweiCloudOBSManager()
files = manager.list_files('backups/models/')
print(f'模型备份文件数量: {len(files)}')
"
```

### 故障排除

#### 常见问题处理
1. **应用无法启动**：
   ```bash
   supervisorctl restart tcm-inspection
   supervisorctl tail -f tcm-inspection
   ```

2. **模型加载失败**：
   ```bash
   # 检查模型文件
   ls -la /home/tcmuser/tcm-inspection/models/
   # 运行模型测试
   python3 test_models.py
   ```

3. **502 Bad Gateway错误**：
   ```bash
   # 检查后端服务
   ps aux | grep gunicorn
   # 重启Nginx
   sudo systemctl restart nginx
   ```

### 性能优化建议

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
- 启用Nginx缓存和压缩
- 配置适当的超时时间
- 模型预加载减少冷启动时间

### 风险控制

#### 技术风险
- **模型文件较大**：使用OBS分片上传，监控上传进度
- **内存使用量高**：配置swap文件，设置内存监控告警
- **API密钥安全**：使用华为云KMS服务管理密钥

#### 运营风险
- **数据隐私合规**：数据本地处理，符合华为云合规要求
- **服务器维护**：配置华为云监控告警，设置自动备份
- **成本控制**：设置费用预警，定期检查资源使用情况

---

## 🚀 部署执行指南

### 快速部署步骤

1. **环境准备**：
   ```bash
   # 上传部署文件到服务器
   scp deploy/* tcmuser@YOUR_ECS_PUBLIC_IP:/home/tcmuser/
   ```

2. **执行主部署脚本**：
   ```bash
   # 在ECS服务器上执行
   cd /home/tcmuser
   chmod +x huaweicloud-deploy.sh
   ./huaweicloud-deploy.sh
   ```

3. **部署AI模型**：
   ```bash
   # 在本地执行模型上传
   bash deploy/model-deployment.sh D:\TCMinspection
   ```

4. **验证部署**：
   ```bash
   # 在服务器上验证
   supervisorctl status tcm-inspection
   python3 test_models.py
   curl http://localhost/health
   ```

### 部署验证清单

- [ ] 网站可以通过公网IP访问
- [ ] 图片上传功能正常
- [ ] AI模型推理正常
- [ ] 智能对话功能正常
- [ ] 检测报告生成正常
- [ ] SSL证书正常工作
- [ ] 监控脚本正常运行
- [ ] 备份功能正常
- [ ] 日志记录正常

### 技术支持

**华为云技术支持**：
- 官方文档：https://support.huaweicloud.com/
- 工单系统：华为云控制台 → "工单" → "提交工单"
- 技术支持电话：950808

**项目文件说明**：
- `deploy/README-HUAWEI-CLOUD.md` - 完整部署文档
- `deploy/huaweicloud-deploy.sh` - 主部署脚本
- `deploy/model-deployment.sh` - 模型部署脚本
- `deploy/huaweicloud-obs-integration.py` - OBS集成脚本
- `deploy/nginx-huaweicloud.conf` - Nginx配置文件
- `deploy/huaweicloud-config.env` - 环境配置模板

每个文件都包含详细的注释和配置说明，确保您能够顺利完成华为云部署。

# Claude MCP Policy
version: 1.0
type: mcp_policy
title: Context7 + Playwright 工作流策略
date: 2025-10-08

---

## 🎯 目标
通过结合 **Context 7** 与 **Playwright**，建立一个智能、可验证的前端开发闭环工作流。  
Claude Code 将在知识检索、任务执行与前端检测三个阶段间自动切换，以实现高效与稳健的协作开发体验。

---

## 🧠 模块定义

### Context 7（知识与指南模块）
**职责：**  
- 提供与项目相关的知识指导、文档检索与开发建议。  
- 充当前端与后端任务的“参考标准”与“知识顾问”。

**触发逻辑：**
1. 当 Claude Code 接收到与前端开发、框架用法、API 或依赖库相关的任务请求时：
   - 优先调用 **Context 7** 检索相关指南；
   - 若找到匹配内容 → Claude 依照文档执行开发；
   - 若无相关文档 → 自动切换至 **Playwright** 以进行动态验证或实测。

**输出要求：**
- 提供准确的技术指导；
- 若无法直接回答，应显式返回空文档信号 `{context7: null}` 以触发下一步 Playwright 测试逻辑。

---

### Playwright（前端检测与执行模块）
**职责：**
- 负责端到端（E2E）交互测试、UI渲染验证与功能行为检查；
- 在任务完成后，对前端表现进行自动检测并生成结果报告。

**触发逻辑：**
1. 当前端任务完成或检测阶段被触发时：
   - 自动启动浏览器实例；
   - 执行以下检测逻辑：
     - 页面加载是否成功；
     - DOM 结构与样式是否符合预期；
     - 按钮/交互组件是否可点击；
     - 控制台日志中是否存在错误；
   - 生成详细测试报告（含截图、错误信息、通过率等）。

2. 若检测通过 → 记录结果并结束任务；
   若检测失败 → 反馈问题报告并调用 Context 7 提供修复建议。

**输出要求：**
- 返回 JSON 格式的检测结果：
  ```json
  {
    "status": "failed",
    "errors": ["Button click not responding", "Missing CSS class .nav-bar"],
    "screenshots": ["screenshot_1.png"]
  }
