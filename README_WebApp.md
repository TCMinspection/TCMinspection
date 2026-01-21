# 中医AI智能望诊系统 - Web应用

## 📋 系统概述

这是一个基于人工智能的中医面色诊断Web应用系统，结合传统中医理论与现代机器学习技术，实现了自动化的面色识别、智能诊断报告生成以及交互式健康咨询功能。

## 🏗️ 系统架构

### 核心模块
1. **图像分析模块** - 基于MindSpore的CNN特征提取和SVM分类
2. **报告生成模块** - 集成LLM的智能中医分析报告
3. **对话咨询模块** - AI健康问答和建议
4. **Web界面模块** - 现代化的响应式前端界面

### 技术栈
- **后端**: Flask + MindSpore + OpenCV + scikit-learn
- **前端**: HTML5 + Tailwind CSS + Vanilla JavaScript
- **AI**: LangChain + OpenAI GPT + FAISS向量检索
- **机器学习**: VGG16特征提取 + SVM分类器 + 自编码器降维

## 🚀 快速开始

### 1. 环境要求
- Python 3.8+
- Windows/Linux/macOS
- 至少8GB内存（推荐16GB）

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 下载必需文件
1. **dlib面部关键点检测器**:
   ```
   # 从以下地址下载
   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

   # 解压后放置到以下目录之一：
   - D:/dlib-predictor/shape_predictor_68_face_landmarks.dat
   - 项目根目录/shape_predictor_68_face_landmarks.dat
   ```

2. **模型文件**（如果不存在）:
   - `autoencoder_model_mindspore.ckpt` - 自编码器模型
   - `svm_pipeline.pkl` - SVM分类器

### 4. 修复导入问题（如果遇到）

#### 方法1：使用新的导入管理系统（推荐）
```bash
# 运行新的模块导入测试
python test_new_import.py

# 如果测试通过，直接启动应用
python app.py
```

#### 方法2：使用传统修复方案
```bash
# 运行导入修复脚本
python fix_imports.py

# 或者手动创建__init__.py文件
touch code.part1/__init__.py
touch code.part2/__init__.py
touch code.part4/__init__.py
```

#### 方法3：详细诊断
```bash
# 运行模块导入管理器进行详细诊断
python module_imports.py

# 运行完整的系统测试
python test_system.py
```

### 5. 启动系统
```bash
# 方法1：使用启动脚本（推荐）
python run.py

# 方法2：直接启动Flask应用
python app.py

# 方法3：先运行测试确保一切正常
python test_new_import.py && python app.py
```

### 6. 访问系统
打开浏览器访问: http://localhost:5000

## 📖 功能说明

### 🔍 图像分析功能
- 支持JPG、PNG、GIF格式图片上传
- 自动检测面部并提取9个关键区域特征
- 基于MindSpore VGG16进行CNN特征提取
- 使用自编码器降维和SVM分类识别面色类型
- 支持5种中医面色分类：红色、黑色、白色、黄色、青色

### 📋 智能报告生成
- 基于识别结果自动生成详细中医分析报告
- 包含面色类型、中医辨证、健康建议
- 集成RAG知识库提供专业中医理论支撑
- 支持个性化患者信息录入

### 💬 AI健康咨询
- 针对不同面色类型提供专业健康建议
- 支持用药、饮食、调理时间、生活方式等咨询
- 集成对话记忆功能，支持多轮连续对话
- 基于LangChain和OpenAI GPT的智能问答

### 🎨 用户界面
- 现代化响应式设计，支持桌面和移动设备
- 直观的拖拽上传界面
- 实时分析进度显示
- 美观的结果展示和报告下载功能

## 🔧 API接口

### 核心接口

#### 1. 系统健康检查
```http
GET /api/health
```

#### 2. 图像分析
```http
POST /api/analyze
Content-Type: multipart/form-data

参数:
- file: 图片文件
- patient_info: 患者信息JSON（可选）
```

#### 3. 生成报告
```http
POST /api/report
Content-Type: application/json

{
    "complexion_type": 0,
    "confidence": 0.85,
    "patient_info": {
        "name": "张三",
        "age": 30,
        "gender": "男"
    }
}
```

#### 4. AI对话
```http
POST /api/chat
Content-Type: application/json

{
    "complexion_type": 0,
    "message": "我最近总是感觉口干舌燥，需要吃什么中药调理？",
    "patient_info": {
        "name": "张三",
        "age": 30,
        "gender": "男"
    },
    "session_id": "session_123"
}
```

#### 5. 获取面色类型
```http
GET /api/complexion_types
```

## 🏥 中医面色分类

### 5种面色类型及其对应中医证型

| 类型 | 颜色 | 中医证型 | 主要特征 |
|------|------|----------|----------|
| 0 | 红色 🔴 | 阴虚火旺 | 面部红赤，多为阴虚内热 |
| 1 | 黑色 ⚫ | 肾虚寒盛 | 面色晦暗，多为肾阳不足 |
| 2 | 白色 ⚪ | 气血两虚 | 面色苍白，多为气血亏虚 |
| 3 | 黄色 🟡 | 脾虚湿盛 | 面色萎黄，多为脾虚湿困 |
| 4 | 青色 🔵 | 寒凝血瘀 | 面色青紫，多为寒凝血瘀 |

## ⚙️ 配置说明

### 环境变量配置
```python
# API密钥配置（在code.part4/agent.py和enhanced_agent.py中）
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Flask配置
FLASK_ENV=development
FLASK_DEBUG=1
FLASK_PORT=5000
```

### 模型文件路径
- VGG16模型: 自动从mindcv加载
- 自编码器: `autoencoder_model_mindspore.ckpt`
- SVM分类器: `svm_pipeline.pkl`
- dlib检测器: `D:/dlib-predictor/shape_predictor_68_face_landmarks.dat`

## 🔍 故障排除

### 常见问题

#### 1. 模型加载失败
```
错误: 模型未完全加载
解决: 检查模型文件是否存在，路径是否正确
```

#### 2. 面部检测失败
```
错误: 未检测到面部
解决: 确保图片清晰，面部正对镜头，光线充足
```

#### 3. API调用失败
```
错误: OpenAI API调用失败
解决: 检查API密钥配置和网络连接
```

#### 4. 内存不足
```
错误: CUDA out of memory
解决: 使用CPU模式或减少批次大小
```

### 调试模式
```bash
# 启用详细日志
export FLASK_DEBUG=1
python app.py
```

## 📊 性能优化

### 硬件要求
- **CPU**: Intel i5或更高
- **内存**: 8GB+（推荐16GB）
- **存储**: 5GB可用空间
- **GPU**: 可选，支持CUDA加速

### 优化建议
1. 使用SSD存储提高模型加载速度
2. 配置GPU加速图像处理
3. 使用Redis缓存分析结果
4. 部署到生产环境时使用Gunicorn

## 🔒 安全注意事项

1. **医疗安全**: 系统生成的建议仅供参考，不能替代专业医师诊断
2. **数据隐私**: 本地处理，不上传患者数据到云端
3. **API安全**: 生产环境部署时需要添加认证和授权机制
4. **输入验证**: 所有用户输入都经过验证和清理

## 📝 更新日志

### v1.0.0 (2024-10-03)
- ✅ 完整的Web应用界面
- ✅ 集成MindSpore图像分析
- ✅ SVM面色分类功能
- ✅ AI报告生成和对话
- ✅ 响应式前端设计
- ✅ RESTful API接口

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进系统！

### 开发环境设置
```bash
git clone <repository>
cd TCMinspection
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

## 📄 许可证

本项目仅供学术研究使用，请勿用于商业医疗诊断。

## 📞 技术支持

如有问题，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**⚠️ 免责声明**: 本系统仅供辅助参考，不能替代专业中医师的诊断。如有健康问题，请及时就医。