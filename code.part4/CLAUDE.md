# CLAUDE.md - 中医智能面色分析系统使用指南

本文件为Claude Code (claude.ai/code) 提供在该代码仓库中工作的指导。

## 项目概述

这是一个**中医（Traditional Chinese Medicine）智能面色分析系统** - 基于人工智能的诊断工具，根据中医原理分析面部面色并生成详细的健康报告和个性化建议。

## 开发命令

### 运行基础TCM分析代理
```bash
python agent.py
```

### 运行增强对话代理
```bash
python enhanced_agent.py
```

### 运行快速分析函数
```bash
python -c "from agent import quick_report; print(quick_report(0, {'name': '张三', 'age': 30, 'gender': '男'}, include_recommendations=True))"
```

### 运行快速聊天诊断
```bash
python -c "from enhanced_agent import quick_chat_diagnosis; print(quick_chat_diagnosis(0, '请问饮食上需要注意什么？', {'name': '张三', 'age': 30, 'gender': '男'}))"
```

## 架构概述

### 核心组件

**1. TCMComplexionAnalysisAgent (agent.py:26-300)**
- 基础中医分析，集成RAG（检索增强生成）功能
- 主要方法：
  - `generate_detection_report()` - 创建正式中医报告
  - `quick_analysis()` - 简化分析功能
  - `save_report()` - 保存报告到./reports目录

**2. TCMEnhancedAgent (enhanced_agent.py:196-960)**
- 高级对话AI，具备记忆管理功能
- 关键类：
  - `ConversationMemory` - 管理会话持久化
  - `IntentDetector` - 识别用户意图（用药、饮食、时间、生活方式、一般咨询）
  - `TCMEnhancedAgent` - 主要对话接口

### 中医面色分类系统

系统识别5种传统模式，存储在`TCM_COMPLEXION_TYPES`中：
- Type 0: 红色 (R) - 阴虚火旺 (阴虚有火)
- Type 1: 黑色 (B) - 肾虚寒盛 (肾虚寒重)
- Type 2: 白色 (W) - 气血两虚 (气血不足)
- Type 3: 黄色 (Y) - 脾虚湿盛 (脾虚湿重)
- Type 4: 青色 (C) - 寒凝血瘀 (寒凝血瘀)

### 关键依赖

```python
# 核心AI/ML技术栈
langchain_openai          # 通过OpenRouter集成OpenAI
langchain                 # AI编排框架
langchain_community       # 社区组件
faiss-cpu                 # 向量相似度搜索
numpy                     # 数值计算
```

### API配置

**安全警告**：API密钥在两个代理文件中都是硬编码的。开发时通过以下方式设置：
```python
os.environ["OPENAI_API_KEY"] = "sk-or-v1-..."  # agent.py第15行，enhanced_agent.py第29行
```

系统使用OpenRouter API和GPT模型：
- 基础URL: `https://openrouter.ai/api/v1`
- 默认模型: `openai/gpt-5`

### 数据流架构

1. **分析流程**：用户输入 → 面色识别 → LLM分析（带RAG）→ 报告生成
2. **对话流程**：用户消息 → 意图识别 → 上下文检索 → 个性化回复 → 记忆更新
3. **RAG系统**：网页爬取 → 文档拆分 → 向量存储（FAISS）→ 上下文检索

### 文件组织

- `/reports/` - 生成的中医分析报告
- 对话记忆存储在`conversation_memory/`目录（动态创建）
- 没有requirements.txt - 依赖关系需要从导入语句推断

### 关键代码模式

**RAG初始化模式**（两个代理）：
```python
def _init_rag_system(self):
    loader = WebBaseLoader(knowledge_urls)
    docs = loader.load_and_split()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever()
```

**报告生成模式**（agent.py:90-156）：
```python
def generate_detection_report(self, complexion_type, image_path, patient_info):
    # 1. 图像分析模拟
    # 2. 带RAG上下文的LLM分析
    # 3. 中文格式报告
    # 4. 建议生成
```

**对话记忆模式**（enhanced_agent.py:63-119）：
```python
# 基于会话的持久化对话（JSON序列化）
# 从对话历史中提取模式用于上下文感知
```

## 重要注意事项

1. **语言**：所有报告和回复都是中文（中医术语）
2. **安全性**：系统包含`needs_doctor_review()`检查以确保医疗安全
3. **无测试**：不存在测试文件 - 需要手动验证更改
4. **无代码检查**：无代码检查配置
5. **API成本**：使用OpenRouter API调用 - 测试时监控使用情况
6. **错误处理**：基础错误处理存在 - 生产环境可能需要增强

## 集成示例（基于代码中的实际使用）

### 使用TCMComplexionAnalysisAgent（来自agent.py的使用示例）

```python
from agent import TCMComplexionAnalysisAgent, TCM_COMPLEXION_TYPES

# 创建代理实例 - 会自动初始化LLM和RAG系统
agent = TCMComplexionAnalysisAgent()

# 快速分析 - 最常用的功能，适合简单场景
# complexion_type: 0=红色 1=黑色 2=白色 3=黄色 4=青色
# confidence: 面色识别的置信度，0.0-1.0之间
# patient_info: 患者基本信息，支持中英文
report = agent.quick_analysis(
    complexion_type=3,  # 黄色 - 脾虚湿盛型
    confidence=0.87,    # 置信度87%
    patient_info={"姓名": "张三", "年龄": "35岁", "性别": "男"}
)
print(report)  # 直接输出分析报告

# 完整详细分析 - 带RAG知识库增强的正式报告
# 需要图像路径（可选），会生成更详细的中医建议
patient_info = {"name": "李四", "age": 28, "gender": "女"}  # 中英文都可以
full_report = agent.generate_detection_report(
    complexion_type=0,      # 红色 - 阴虚火旺型
    image_path="patient_face.jpg",  # 图片路径，传入None也可以纯文本分析
    patient_info=patient_info
)

# 保存报告到文件 - 自动保存到./reports目录
# 可自定义文件名，不传则用时间戳自动生成
report_file = agent.save_report(full_report, "my_tcm_report.txt")
print(f"报告已保存至: {report_file}")  # 返回完整的文件路径
# 不传文件名则自动生成: tcm_complexion_report_20250928_133630.txt
```

### 使用TCMEnhancedAgent进行对话（来自enhanced_agent.py的使用示例）

```python
from enhanced_agent import quick_chat_diagnosis

# 准备常见的问题列表 - 这些是基于中医理论的典型咨询问题
# 任何中医相关的问题都可以：饮食、用药、调理时间、生活方式等
test_questions = [
    "我最近总是感觉口干舌燥，需要吃什么中药调理？",
    "这种面色饮食上有什么需要特别注意的吗？",
    "大概需要调理多长时间才能看到改善？",
    "平时生活和运动上要注意什么？"
]

# quick_chat_diagnosis - 最简单直接的对话方式
# complexion_type: 必须先确定面色类型(0-4)
# user_question: 用户的问题，可以是任意中医相关内容
# patient_info: 基本信息，name和age是必须的
for question in test_questions:
    response = quick_chat_diagnosis(
        complexion_type=0,  # 红色 - 阴虚火旺型
        user_question=question,  # 用户问题
        patient_info={"name": "李女士", "age": 35}  # 基本信息
    )
    print(f"用户：{question}")
    print(f"AI：{response}")
    # response返回的是完整的中文回答，基于RAG知识库

# TCMEnhanceAgent - 完整的对话代理，支持多轮对话和上下文记忆
from enhanced_agent import TCMEnhancedAgent

# 创建代理实例，可以配置各种参数
chat_agent = TCMEnhancedAgent({"model_name": "openai/gpt-5"})

# handle_user_message - 完整功能的对话接口
# session_id: 会话ID，同一会话会自动记忆之前的对话内容
# diagnosis_result: 必须包含complexion_type和confidence
# patient_info: 患者详细信息，越详细回复越个性化
response = chat_agent.handle_user_message(
    user_message="我最近总是感觉疲劳，有什么调理建议？", # 用户问题
    session_id="patient_session_123",  # 会话ID，用于记忆功能
    complexion_type=2,  # 白色 - 气血两虚型
    patient_info={"name": "王五", "age": 45, "gender": "男"},  # 患者信息
    diagnosis_result={
        "complexion_type": 2,  # 面色类型编码
        "confidence": 0.8,     # 诊断置信度
        "symptoms": ["疲劳", "面色苍白"]  # 症状列表，用于个性化推荐
    }
)

# 返回结果包含：
# - response: AI的回答内容
# - intent_category: 识别的用户意图（medication/diet/timeline/lifestyle/general)
# - needs_doctor_review: 是否需要医生人工审核
# - session_id: 会话ID，用于下次对话

### 关键集成模式（基于代码示例）

**1. TCMComplexionAnalysisAgent Pattern** (from line 332-336 in agent.py)
```python
# 最基础的快速分析模式 - 适合批量处理和接口集成
agent = TCMComplexionAnalysisAgent()  # 创建代理实例，仅一次即可

# quick_analysis - 最快速简单的调用方式
# 参数说明：
# complexion_type: 面色类型，必须传入0-4的整数（对应中医的5种面色）
# confidence: 可选，面色识别的置信度，范围0.0-1.0，影响报告中的置信度说明
# patient_info: 患者信息字典，支持中英文键名
result = agent.quick_analysis(
    complexion_type=complexion_type,  # 0=红色 1=黑色 2=白色 3=黄色 4=青色
    confidence=confidence_score,      # 如：0.85 表示85%的置信度
    patient_info=patient_dict         # {'name':'张三'} 或 {'姓名':'张三'}都可以
)
print(result)  # 返回标准格式的中医分析报告字符串

# 可选的参数组合：
# agent.quick_analysis(0, 0.9, {"name":"张三"})  # 极简调用
# agent.quick_analysis(2, None, patient_info)   # 不包含信度
# agent.quick_analysis(3, 0.75, {"姓名":"李四","年龄":30})  # 包含更多患者信息
```

**2. TCMEnhancedAgent Pattern** (from line 1004 in enhanced_agent.py)
```python
# 最简单直接的对话模式 - 适合快速生成答案，无需管理状态
# 适用于：单次问答、简单咨询、快速建议
response = quick_chat_diagnosis(
    complexion_type=0,                    # 面色类型（0-4），必须先确定
    user_question=user_question,          # 用户问题文本，任意中医相关问题
    patient_info={"name": "姓名", "age": 35}  # 基本信息必须有name和age
)

# quick_chat_diagnosis特点：
# ✅ 无需创建代理实例，一行代码调用
# ✅ 自动处理RAG检索（如可用）
# ✅ 返回完整中文回答
# ✅ 支持所有常见问题类型
# ❌ 无状态记忆，每调用都是独立的
```

**3. Session-based Conversation Pattern** (from enhanced_agent conversation flow)
```python
# 完整的对话代理模式 - 支持多轮对话和上下文记忆
from enhanced_agent import TCMEnhancedAgent

# 创建对话代理实例 - 可自定义配置参数
chat_agent = TCMEnhancedAgent({
    "model_name": "openai/gpt-5",           # LLM模型选择
    "memory_capacity": 50,                 # 记忆容量限制
    "enable_rag": True                     # 是否启用RAG增强
})

# handle_user_message - 完整功能接口
# 适用于：多轮对话、个性化推荐、复杂问诊场景
response_dict = chat_agent.handle_user_message(
    user_message="问题内容",                            # 用户问题（支持中文）
    session_id="unique_session_id",                    # 会话ID，承载记忆功能
    complexion_type=complexion_type,                   # 面色类型（0-4）
    patient_info=patient_info,                         # 患者详细信息
    diagnosis_result=diagnosis_dict                    # 诊断结果上下文
)

# 返回字典结构说明：
# response_dict['response'] - AI的完整回答内容
# response_dict['intent_category'] - 用户意图分类
# response_dict['needs_doctor_review'] - 是否需要医生审核
# response_dict['session_id'] - 会话ID，用于下次对话
```