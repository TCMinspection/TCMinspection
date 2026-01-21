#!/usr/bin/env python3
"""
TCM智能对话代理 - 增强版
集成LLM、RAG、对话记忆和上下文感知的中医调理咨询系统
"""


import json
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from dataclasses import dataclass, asdict

# LangChain组件
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

os.environ["OPENAI_API_KEY"] = "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"

# TCM定义
TCM_COMPLEXION_TYPES = {
    0: {"name": "红色", "icon": "🔴", "tcm_type": "阴虚火旺", "yinyang": "阴虚", "color": "#e74c3c"},
    1: {"name": "黑色", "icon": "⚫", "tcm_type": "肾虚寒盛", "yinyang": "阳虚", "color": "#2c3e50"},
    2: {"name": "白色", "icon": "⚪", "tcm_type": "气血两虚", "yinyang": "阳虚", "color": "#bdc3c7"},
    3: {"name": "黄色", "icon": "🟡", "tcm_type": "脾虚湿盛", "yinyang": "偏虚", "color": "#f39c12"},
    4: {"name": "青色", "icon": "🔵", "tcm_type": "寒凝血瘀", "yinyang": "寒证", "color": "#3498db"}
}

@dataclass
class ConversationContext:
    """对话上下文数据结构"""
    session_id: str
    patient_info: Dict[str, Any]
    diagnosis_result: Dict[str, Any]
    complexion_type: int
    confidence: float
    conversation_history: List[Dict[str, Any]]
    is_follow_up: bool
    message_count: int
    created_at: datetime
    last_activity: datetime

class TCMMemoryManager:
    """TCM对话记忆管理器"""

    def __init__(self, memory_dir: str = "conversation_memory"):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        self.memories = {}
        self.max_memory_per_session = 100

    def save_conversation_memory(self, session_id: str, conversation_data: Dict[str, Any]):
        """保存对话记忆"""
        memory_file = os.path.join(self.memory_dir, f"{session_id}_memory.json")

        # 加载现有记忆或创建新的
        if os.path.exists(memory_file):
            with open(memory_file, 'r', encoding='utf-8') as f:
                existing_memory = json.load(f)
        else:
            existing_memory = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "conversations": [],
                "patterns": {},
                "total_interactions": 0
            }

        # 添加新的对话记录
        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "diagnosis_context": conversation_data.get("diagnosis_context", {}),
            "user_messages": conversation_data.get("user_messages", []),
            "ai_responses": conversation_data.get("ai_responses", []),
            "intent_categories": conversation_data.get("intent_categories", []),
            "effectiveness_rating": conversation_data.get("effectiveness_rating"),
            "user_feedback": conversation_data.get("user_feedback")
        }

        existing_memory["conversations"].append(conversation_entry)
        existing_memory["total_interactions"] += 1

        # 保持记忆大小在合理范围内
        if len(existing_memory["conversations"]) > self.max_memory_per_session:
            existing_memory["conversations"] = existing_memory["conversations"][-self.max_memory_per_session:]

        # 保存记忆
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(existing_memory, f, ensure_ascii=False, indent=2)

        # 更新内存中的记忆
        self.memories[session_id] = existing_memory

    def load_conversation_memory(self, session_id: str) -> Dict[str, Any]:
        """加载对话记忆"""
        if session_id in self.memories:
            return self.memories[session_id]

        memory_file = os.path.join(self.memory_dir, f"{session_id}_memory.json")
        if os.path.exists(memory_file):
            with open(memory_file, 'r', encoding='utf-8') as f:
                memory = json.load(f)
                self.memories[session_id] = memory
                return memory

        return None

    def extract_patterns(self, session_id: str) -> Dict[str, Any]:
        """从对话记忆中提取模式"""
        memory = self.load_conversation_memory(session_id)
        if not memory or not memory.get("conversations"):
            return {}

        patterns = {
            "most_asked_topics": {},
            "common_concerns": [],
            "preferred_advice_types": {},
            "session_duration_pattern": [],
            "user_engagement_level": "medium"
        }

        # 统计最常见的问题主题
        topic_counts = {}
        for conv in memory["conversations"]:
            for intent in conv.get("intent_categories", []):
                topic_counts[intent] = topic_counts.get(intent, 0) + 1

        patterns["most_asked_topics"] = dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5])

        # 统计会话时长模式
        if len(memory["conversations"]) >= 2:
            durations = []
            for i in range(1, len(memory["conversations"])):
                prev_time = datetime.fromisoformat(memory["conversations"][i-1]["timestamp"])
                curr_time = datetime.fromisoformat(memory["conversations"][i]["timestamp"])
                durations.append((curr_time - prev_time).total_seconds() / 60)  # 分钟
            patterns["session_duration_pattern"] = durations

        return patterns

class TCMIntentDetector:
    """TCM意图检测器"""

    def __init__(self):
        self.intent_patterns = {
            "medication": ["药", "方", "剂", "汤", "丸", "处方", "吃药", "用药"],
            "diet": ["饮食", "食物", "吃", "忌口", "食疗", "食谱", "能喝", "能吃"],
            "lifestyle": ["作息", "运动", "生活", "起居", "锻炼", "睡觉", "休息"],
            "timeline": ["多久", "时间", "见效", "周期", "疗程", "几天", "几周"],
            "symptoms": ["症状", "表现", "感觉", "不舒服", "难受", "疼痛"],
            "causes": ["原因", "为什么", "怎么引起", "病因", "病机"],
            "prevention": ["预防", "注意", "避免", "防止", "恶化", "加重"]
        }

    def detect_intent(self, message: str) -> List[str]:
        """检测用户消息意图"""
        message_lower = message.lower()
        detected_intents = []

        for intent, keywords in self.intent_patterns.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_intents.append(intent)

        return detected_intents if detected_intents else ["general"]

    def needs_doctor_review(self, intents: List[str], message: str) -> bool:
        """判断是否需要医生审核"""
        # 涉及用药建议的需要医生审核
        if "medication" in intents:
            return True

        # 涉及严重症状的需要医生审核
        if any(word in message for word in ["严重", "危险", "紧急", "急救", "医院"]):
            return True

        # 涉及特殊人群的需要医生审核
        if any(word in message for word in ["孕妇", "哺乳期", "儿童", "老人"]):
            return True

        return False

class TCMEnhancedAgent:
    """TCM增强对话代理 - 完整版"""

    def __init__(self, config: Dict[str, Any]):
        """初始化增强代理"""
        self.config = config
        self.model_name = config.get('model_name', 'openai/gpt-4')
        self.api_base = config.get('api_base', 'https://openrouter.ai/api/v1')
        self.memory_capacity = config.get('memory_capacity', 50)
        self.conversation_timeout = config.get('conversation_timeout', 1800)

        # 初始化LLM
        self._init_llm()

        # 初始化RAG系统
        self._init_rag_system()

        # 初始化记忆管理器
        self.memory_manager = TCMMemoryManager()

        # 初始化意图检测器
        self.intent_detector = TCMIntentDetector()

        # 会话存储
        self.conversation_contexts = {}

        print("✅ TCM增强对话代理初始化完成")

    def _init_llm(self):
        """初始化语言模型"""
        try:
            self.llm = ChatOpenAI(
                model=self.model_name,
                base_url=self.api_base,
                temperature=0.7,
                max_tokens=2000
            )
            print(f"✅ LLM模型初始化完成: {self.model_name}")
        except Exception as e:
            print(f"⚠️ LLM初始化失败: {e}")
            self.llm = None

    def _init_rag_system(self):
        """初始化RAG知识库系统"""
        self.retriever = None
        self.vectorstore = None

        if not self.config.get('enable_rag', True):
            print("RAG系统已禁用")
            return

        knowledge_urls = self.config.get('knowledge_urls', [
            'https://www.scm.cuhk.edu.hk/en-gb/articles-redirect/236-v1-hkej/1767-2020-10-02',
        ])

        try:
            print("🔄 初始化RAG知识库...")

            # 加载文档
            loader = WebBaseLoader(
                web_paths=knowledge_urls,
                header_template={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            docs = loader.load()

            # 文本切分
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
            )
            splits = text_splitter.split_documents(docs)

            # 向量化（使用轻量级模型）
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
            except:
                print("使用OpenAI Embeddings")
                embeddings = OpenAIEmbeddings(base_url=self.api_base)

            self.vectorstore = FAISS.from_documents(splits, embeddings)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

            print("✅ RAG系统初始化成功")

        except Exception as e:
            print(f"⚠️ RAG系统初始化失败: {e}")
            self.retriever = None
            self.vectorstore = None

    def generate_formal_report(self, diagnosis_result: Dict[str, Any], patient_info: Dict[str, Any]) -> str:
        """生成正式检测报告"""
        complexion_type = diagnosis_result.get('complexion_type', 0)
        confidence = diagnosis_result.get('confidence', 0.0)
        info = TCM_COMPLEXION_TYPES.get(complexion_type, TCM_COMPLEXION_TYPES[0])

        # 获取RAG知识增强
        rag_knowledge = self._get_rag_context(f"面色{info['name']}的中医诊断和治疗原则")

        report_prompt = f"""
        您是TCM中医专家，请基于以下检测结果生成专业的面色诊断报告：

        患者信息：{json.dumps(patient_info, ensure_ascii=False)}
        面色类型：{info['name']} ({info['tcm_type']})
        置信度：{confidence:.1%}

        相关知识：
        {rag_knowledge}

        请生成一份包含以下内容的完整报告：
        1. 基本信息和检测结果概述
        2. 中医辨证分析和病理机制解释
        3. 具体的中药调理方案（含方剂和用法）
        4. 详细的饮食调理指导（含推荐食材和禁忌）
        5. 生活方式调整建议（作息、运动、情绪）
        6. 调理效果预期时间表
        7. 重要注意事项和预警信号

        要求：
        - 专业准确，符合中医理论
        - 具体可操作，不要空泛
        - 条理清晰，层次分明
        - 体现个体化辨证思想
        """

        try:
            response = self.llm.invoke(report_prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"报告生成失败: {e}")
            return self._generate_fallback_report(diagnosis_result, patient_info)

    def handle_user_message(self,
                        session_id: str,
                        user_message: str,
                        conversation_context: Dict[str, Any],
                        additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理用户消息"""

        # 提取诊断上下文
        diagnosis_result = conversation_context.get('diagnosis_result', {})
        patient_info = conversation_context.get('patient_info', {})
        complexion_type = diagnosis_result.get('complexion_type', 0)
        confidence = diagnosis_result.get('confidence', 0.0)

        # 检测用户意图
        intents = self.intent_detector.detect_intent(user_message)
        needs_review = self.intent_detector.needs_doctor_review(intents, user_message)

        # 创建对话上下文
        # 确保additional_context有安全默认值
        additional_context = additional_context or {}

        context = ConversationContext(
            session_id=session_id,
            patient_info=patient_info,
            diagnosis_result=diagnosis_result,
            complexion_type=complexion_type,
            confidence=confidence,
            conversation_history=conversation_context.get('conversation_history', []),
            is_follow_up=additional_context.get('is_follow_up', False),
            message_count=additional_context.get('message_count', 0),
            created_at=datetime.now(),
            last_activity=datetime.now()
        )

        # 生成个性化回复
        response = self._generate_personalized_response(
            user_message, context, intents, needs_review
        )

        # 保存记忆 - 处理datetime序列化问题
        memory_data = {
            "user_messages": [user_message],
            "ai_responses": [response],
            "intent_categories": intents,
            "diagnosis_context": self._serialize_context_for_json(context),
            "timestamp": datetime.now().isoformat()
        }

        self.memory_manager.save_conversation_memory(session_id, memory_data)

        return {
            'response': response,
            'intent_category': intents[0] if intents else 'general',
            'needs_doctor_review': needs_review,
            'session_id': session_id
        }

    def _generate_personalized_response(self,
                                    user_message: str,
                                    context: ConversationContext,
                                    intents: List[str],
                                    needs_review: bool) -> str:
        """生成个性化回复"""

        complexion_info = TCM_COMPLEXION_TYPES.get(context.complexion_type, TCM_COMPLEXION_TYPES[0])
        patient_name = context.patient_info.get('name', '您好')

        # 获取RAG知识
        rag_context = ""
        if self.retriever:
            query_topics = [f"面色{complexion_info['name']}", f"{complexion_info['tcm_type']}"]
            for topic in query_topics:
                if topic in user_message or len(intents) == 1:
                    rag_context += self._get_rag_context(f"{topic} {user_message}") + "\n"

        # 构建个性化提示
        personalized_prompt = f"""
        您是资深的TCM中医专家，正在与患者进行深度调理咨询对话。

        【患者背景信息】
        姓名：{patient_name}
        面色诊断：{complexion_info['name']} ({complexion_info['tcm_type']})
        置信度：{context.confidence:.1%}
        阴阳属性：{complexion_info['yinyang']}

        【用户问题】
        {user_message}

        【检测到的意图】
        {', '.join(intents)}

        【专业知识参考】
        {rag_context}

        【对话历史概览】
        这是第{context.message_count}轮对话
        {'后续咨询' if context.is_follow_up else '初始咨询'}

        【重要提示】
        请严格按照系统检测的面色类型进行诊断：
        - 红色 = 阴虚火旺型 (以滋阴降火为主)
        - 黑色 = 肾虚寒盛型 (以温补肾阳为主)
        - 白色 = 气血两虚型 (以补气养血为主)
        - 黄色 = 脾虚湿盛型 (以健脾祛湿为主)
        - 青色 = 寒凝血瘀型 (以温经散寒为主)

        【回复要求】
        1. 严格按照系统检测的面色类型进行辨证，不要混合其他中医理论
        2. 基于用户的具体面色类型({complexion_info['tcm_type']})提供针对性建议
        3. 结合检测到的意图给出个性化回答
        4. 体现中医整体观念和辨证论治思想
        5. 语言要专业但易懂，条理清晰
        6. 必要时提醒用户及时就医

        请提供详细、实用、安全的中医调理建议：
        """

        try:
            if self.llm:
                response = self.llm.invoke(personalized_prompt)
                ai_response = response.content if hasattr(response, 'content') else str(response)
            else:
                ai_response = self._generate_fallback_response(user_message, complexion_info, intents)

            # 添加审核提醒
            if needs_review:
                ai_response = f"""
{ai_response}

⚠️ **温馨提示**
以上建议涉及处方用药，请务必咨询专业中医师后使用，切勿自行配药服用。
"""

            return ai_response

        except Exception as e:
            print(f"个性化回复生成失败: {e}")
            return self._generate_fallback_response(user_message, complexion_info, intents)

    def _get_rag_context(self, query: str) -> str:
        """获取RAG上下文"""
        if not self.retriever:
            return ""

        try:
            docs = self.retriever.invoke(query)
            if docs:
                return "\n".join([doc.page_content for doc in docs[:2]])
        except Exception as e:
            print(f"RAG查询失败: {e}")

        return ""

    def _generate_fallback_response(self, user_message: str, complexion_info: Dict[str, Any], intents: List[str]) -> str:
        """生成回退回复（当LLM不可用时）"""

        complexion_type = list(TCM_COMPLEXION_TYPES.keys())[list(TCM_COMPLEXION_TYPES.values()).index(complexion_info)]

        # 基于意图的回退回复
        if "medication" in intents:
            return self._fallback_medication_response(complexion_type)
        elif "diet" in intents:
            return self._fallback_diet_response(complexion_type)
        elif "timeline" in intents:
            return self._fallback_timeline_response(complexion_type)
        elif "lifestyle" in intents:
            return self._fallback_lifestyle_response(complexion_type)
        else:
            return self._fallback_general_response(complexion_type)

    def _fallback_medication_response(self, complexion_type: int) -> str:
        """用药建议回退回复"""
        responses = {
            0: """💊 **中药调理方案**

**推荐方剂**：知柏地黄丸、麦味地黄丸
**主要功效**：滋阴降火，适用于阴虚火旺体质
**使用注意**：
• 饭后半小时温水送服
• 避免同时食用辛辣刺激性食物
• 连续服用2-4周后评估效果

**配合建议**：可搭配百合银耳羹食疗，加强滋阴效果。""",

            1: """💊 **中药调理方案**

**推荐方剂**：金匮肾气丸、右归丸
**主要功效**：温补肾阳，适用于肾阳虚体质
**使用注意**：
• 空腹或饭前服用效果更佳
• 服药期间注意保暖，避免寒凉
• 需要连续服用1-3个月

**配合建议**：可搭配羊肉枸杞汤食补，增强温补效果。""",

            2: """💊 **中药调理方案**

**推荐方剂**：八珍汤、归脾丸
**主要功效**：补气养血，适用于气血两虚体质
**使用注意**：
• 可长期服用，安全性好
• 最佳服用时间为早晚餐前
• 建议配合充足睡眠

**配合建议**：搭配红枣山药汤食补，气血双补。""",

            3: """💊 **中药调理方案**

**推荐方剂**：参苓白术散、香砂六君子汤
**主要功效**：健脾祛湿，适用于脾虚湿盛体质
**使用注意**：
• 饭前服用，有助于药物吸收
• 服药期间饮食宜清淡
• 避免生冷油腻食物

**配合建议**：可搭配山药小米粥食补，健脾养胃。""",

            4: """💊 **中药调理方案**

**推荐方剂**：逍遥丸、柴胡疏肝散
**主要功效**：疏肝理气，适用于肝郁气滞伴寒凝血瘀体质
**使用注意**：
• 情绪波动大时效果更明显
• 可配合玫瑰花茶代茶饮
• 注意情绪调节

**配合建议**：可搭配玫瑰花陈皮茶，疏肝解郁。"""
        }

        return responses.get(complexion_type, "建议咨询专业中医师进行个体化辨证用药。")

    def _fallback_diet_response(self, complexion_type: int) -> str:
        """饮食建议回退回复"""
        responses = {
            0: """🍲 **饮食调理指导**

**推荐食材**：
• 百合、银耳、绿豆、鸭肉、雪梨、西瓜、苦瓜
• 绿叶蔬菜、冬瓜、丝瓜等清凉性食物

**推荐食疗**：
• 百合银耳羹：滋养肺胃，清热润燥
• 绿豆薏米汤：清热解毒，祛湿健脾

**需要避免**：
• 辣椒、生姜、羊肉、酒类等温热性食物
• 烧烤、油炸等上火食品
• 过于辛辣刺激的调味品""",

            1: """🍲 **饮食调理指导**

**推荐食材**：
• 羊肉、肉桂、核桃、枸杞、黑芝麻、黑豆
• 韭菜、虾仁、桂圆、红枣等温补食材

**推荐食疗**：
• 肉桂羊肉汤：温补肾阳，驱寒暖身
• 核桃枸杞粥：补肾益精，强健腰膝

**需要避免**：
• 生冷寒凉食物：冰淇淋、冷饮、西瓜
• 过于油腻难消化的食物
• 绿豆、苦瓜等寒凉性食材""",

            2: """🍲 **饮食调理指导**

**推荐食材**：
• 红枣、山药、瘦肉、鸡蛋、牛奶、桂圆
• 胡萝卜、菠菜、南瓜等富含营养的食物

**推荐食疗**：
• 红枣山药汤：补脾生血，增强体质
• 当归生姜羊肉汤：补血祛寒，温补气血

**需要避免**：
• 过度生冷的食物和饮料
• 辛辣刺激性强的食物
• 过度油腻影响消化的食物""",

            3: """🍲 **饮食调理指导**

**推荐食材**：
• 山药、薏米、扁豆、莲子、小米、南瓜
• 白萝卜、冬瓜、茯苓等健脾利湿食材

**推荐食疗**：
• 山药薏米扁豆粥：健脾祛湿，消补并重
• 莲子小米粥：健脾养胃，固摄止泻

**需要避免**：
• 过于油腻的食物：油炸食品、肥肉
• 甜腻食物：奶油蛋糕、甜饮料
• 生冷食物：生鱼片、冰镇饮料""",

            4: """🍲 **饮食调理指导**

**推荐食材**：
• 陈皮、玫瑰花、山楂、红糖、桂圆、当归
• 生姜、桂皮、小茴香等温性调料

**推荐食疗**：
• 玫瑰花陈皮茶：疏肝理气，解郁和血
• 当归桂圆红枣汤：活血养血，调经止痛

**需要避免**：
• 生冷食物：冰品、寒性水果
• 过于油腻的食物
• 过于辛辣的刺激性调料"""
        }

        return responses.get(complexion_type, "建议均衡饮食，多吃新鲜蔬果，少食生冷油腻。")

    def _fallback_timeline_response(self, complexion_type: int) -> str:
        """时间安排回退回复"""
        responses = {
            0: """⏰ **调理效果时间预期**

**第1-2周**：
• 燥热感开始减轻，夜间盗汗减少
• 口干咽燥症状有所改善
• 睡眠质量开始提升

**第3-4周**：
• 整体精神状态明显好转
• "上火"症状（如口腔溃疡）减轻
• 情绪稳定性增强

**第6-8周**：
• 基本症状显著改善
• 体质开始趋于平衡
• 面色逐渐恢复正常

**2-3个月以上**：
• 体质明显改善，阴阳平衡
• 精力充沛，免疫力提升
• 复发频率显著降低

✅ **判断调理有效的标准**：
• 睡眠质量改善且稳定
• 精神状态良好，不易疲劳
• 相关不适症状明显减轻""",

            1: """⏰ **调理效果时间预期**

**第1-3周**：
• 畏寒怕冷感开始减轻
• 手脚冰凉状况改善
• 夜间小便次数减少

**第4-6周**：
• 腰膝酸痛逐渐缓解
• 腰背肌肉力量增强
• 行动灵活度明显提升

**第2-3个月**：
• 肾阳充足证候改善
• 性功能有所恢复
• 精力水平显著提升

**第3-6个月以上**：
• 肾阳充足，整体体质增强
• 面色红润有光泽
• 耐寒能力显著增强

✅ **判断调理有效的标准**：
• 手脚温暖，不畏寒
• 腰膝有力，活动自如
• 精神状态饱满""",

            2: """⏰ **调理效果时间预期**

**第2-3周**：
• 体力开始改善，不容易疲劳
• 记忆力有所改善
• 食欲增加，消化功能增强

**第4-6周**：
• 面色苍白开始转为红润
• 口唇颜色变得红润
• 指甲变得饱满有光泽

**第2-3个月**：
• 气血充盈，体质增强
• 面色红润有光泽
• 精神饱满，体力充沛

**第4-6个月以上**：
• 气血充足，体质强健
• 面如桃花，精神焕发
• 免疫力显著提升

✅ **判断调理有效的标准**：
• 面色红润有光泽
• 精力充沛，不易疲劳
• 肢体温暖有力""",

            3: """⏰ **调理效果时间预期**

**第1-2周**：
• 胃纳转佳，食欲明显增加
• 大便开始成形，次数规律
• 消化功能明显改善

**第3-4周**：
• 白天困倦感明显减轻
• 体力恢复，精神好转
• 肢体沉重感减少

**第6-8周**：
• 脾胃功能明显增强
• 体重趋于正常
• 身体感到轻松有力

**第2-3个月以上**：
• 脾胃健运，体质强壮
• 面色红润，精神状态佳
• 肢体有力，活动轻松

✅ **判断调理有效的标准**：
• 食欲良好，消化正常
• 体重稳定，精力充沛
• 肢体轻盈有力""",

            4: """⏰ **调理效果时间预期**

**第2-3周**：
• 气机开始变得舒畅
• 情绪波动有所改善
• 胸部闷胀感减轻

**第4-5周**：
• 面色青紫开始变淡
• 肢体疼痛感减轻
• 经期不适感减少

**第6-10周**：
• 瘀血开始化散
• 气血运行明显改善
• 面色趋于正常

**第3-6个月以上**：
• 瘀血化散透彻
• 气血畅通无阻
• 情绪稳定，容颜焕彩

✅ **判断调理有效的标准**：
• 面色正常红润
• 情绪平稳愉悦
• 肢体灵活无痛"""
        }

        return responses.get(complexion_type, "体质调理通常需要1-3个月的持续改善过程。")

    def _fallback_lifestyle_response(self, complexion_type: int) -> str:
        """生活方式回退回复"""
        responses = {
            0: """🧘‍♀️ **生活方式指导**

**作息建议**：
• 保持充足睡眠，每晚7-8小时
• 避免经常熬夜，最好在11点前入睡
• 午休不宜过长，20-30分钟为宜

**运动建议**：
• 选择舒缓运动：太极拳、瑜伽、游泳
• 避免长时间剧烈运动，以免耗伤阴液
• 运动强度以微微出汗为宜
• 最佳运动时间：傍晚时分

**情绪管理**：
• 保持心情愉悦，避免情绪激动
• 学会情绪自我调节方法
• 避免过度焦虑和烦躁
• 可尝试冥想或深呼吸放松""",

            1: """🧘‍♀️ **生活方式指导**

**作息建议**：
• 早睡早起，保证睡眠质量
• 睡前泡脚15-20分钟，温养肾阳
• 注意保暖，特别是腰部和脚部
• 避免长时间在空调环境中

**运动建议**：
• 温和运动：慢跑、快走、八段锦
• 太极拳等温和的传统运动
• 运动强度适中，避免过度劳累
• 运动后及时保暖，避免着凉

**情绪管理**：
• 保持积极乐观的心态
• 多参与社交和户外活动
• 避免情绪低落和忧郁
• 培养兴趣爱好，转移注意力""",

            2: """🧘‍♀️ **生活方式指导**

**作息建议**：
• 保证充足睡眠，每日8小时以上
• 午间适当休息，恢复体力
• 避免夜间加班或熬夜
• 疲劳时及时休息，不要硬撑

**运动建议**：
• 轻度运动：散步、伸展运动
• 太极拳、气功等柔和运动
• 运动时间和强度要逐步增加
• 避免突然的高强度运动

**情绪管理**：
• 保持开朗乐观的心态
• 避免过度思虑伤气血
• 多与亲友交流，获得支持
• 培养健康兴趣爱好，丰富生活""",

            3: """🧘‍♀️ **生活方式指导**

**作息建议**：
• 规律饮食，定时定量
• 午餐后散步10-15分钟助消化
• 避免饭后立即卧床休息
• 睡前2小时内不进食

**运动建议**：
• 饭后散步有助脾胃运化
• 太极拳、八段锦等传统养生术
• 避免饭后立即剧烈活动
• 保持适量运动，不宜过劳

**情绪管理**：
• 保持心情舒畅，避免郁结
• 凡事不急不躁，顺其自然
• 多参与轻松愉快的活动
• 学会情绪调节和放松方法""",

            4: """🧘‍♀️ **生活方式指导**

**作息建议**：
• 注意保暖，避免受寒着凉
• 适当进行温热泡脚
• 避免长时间接触冷水
• 冬季尤其要做好保暖工作

**运动建议**：
• 瑜伽、太极等柔和舒展运动
• 促进气血流通的有氧运动
• 避免久坐久卧，要适度活动
• 运动后注意保暖，防止受凉

**情绪管理**：
• 保持情绪平和稳定
• 避免生气发怒，防止气机郁滞
• 多与亲朋好友交流沟通
• 培养豁达乐观的生活态度
• 可适当进行按摩放松"""
        }

        return responses.get(complexion_type, "保持规律作息，适度运动，调畅情志。")

    def _fallback_general_response(self, complexion_type: int) -> str:
        """通用回退回复"""
        complexion_info = TCM_COMPLEXION_TYPES.get(complexion_type, TCM_COMPLEXION_TYPES[0])

        return f"""
基于您的{complexion_info['name']}面色特征，这里是综合调理建议：

🎯 **主要调理方向**：
• 证型特点：{complexion_info['tcm_type']}
• 体质类型：{complexion_info['yinyang']}体质
• 调理原则：{self._get_tcm_principle(complexion_type)}

💡 **日常生活要点**：
• 作息：保持规律作息，充足睡眠
• 饮食：选择适合自己体质的食物
• 运动：适度运动，循序渐进
• 情志：保持心情舒畅，避免过度情绪波动

🗣️ **您可以继续询问**：
• 具体的中药调理方案
• 详细的饮食搭配和食谱
• 运动强度和频率建议
• 调理过程中的注意事项
• 症状变化的预期时间

还有什么具体问题需要详细解答吗？我会结合您的面色诊断为您提供更精准的建议。"""

    def _get_tcm_principle(self, complexion_type: int) -> str:
        """获取中医调理原则"""
        principles = {
            0: "滋阴降火，清热润燥",
            1: "温补肾阳，祛寒除湿",
            2: "补气养血，温阳健脾",
            3: "健脾祛湿，调和气血",
            4: "温经散寒，疏肝理气"
        }
        return principles.get(complexion_type, "辨证施治，整体调理")

    def _serialize_context_for_json(self, context: ConversationContext) -> Dict[str, Any]:
        """将对话上下文序列化为可JSON化的字典"""
        context_dict = asdict(context)
        # 将datetime对象转换为可序列化的字符串
        context_dict['created_at'] = context.created_at.isoformat()
        context_dict['last_activity'] = context.last_activity.isoformat()
        return context_dict

    def _generate_fallback_report(self, diagnosis_result: Dict[str, Any], patient_info: Dict[str, Any]) -> str:
        """生成回退检测报告"""
        complexion_type = diagnosis_result.get('complexion_type', 0)
        confidence = diagnosis_result.get('confidence', 0.0)
        info = TCM_COMPLEXION_TYPES.get(complexion_type, TCM_COMPLEXION_TYPES[0])

        return f"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                    TCM面色检测报告                    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

【基本信息】
检测时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
患者: {patient_info.get('name', '匿名')} {patient_info.get('age', '')} {patient_info.get('gender', '')}

【诊断结果】
面色类型: {info['name']} ({info['tcm_type']})
置信度: {confidence:.1%}

【中医辨证】
主要病机: {self._get_tcm_principle(complexion_type)}
体质特点: {info['yinyang']}体质

【基础调理建议】
{self._fallback_general_response(complexion_type)}

⚠️ 本报告由AI辅助系统生成，仅供参考。具体诊疗请咨询专业中医师。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

# 快速使用函数
def quick_chat_diagnosis(complexion_type: int, user_question: str, patient_info: Dict[str, Any] = None):
    """快速对话诊断函数"""
    config = {
        'model_name': 'openai/gpt-4',
        'api_base': 'https://openrouter.ai/api/v1',
        'enable_rag': True
    }

    agent = TCMEnhancedAgent(config)

    diagnosis_result = {
        'complexion_type': complexion_type,
        'confidence': 0.85
    }

    conversation_context = {
        'diagnosis_result': diagnosis_result,
        'patient_info': patient_info or {},
        'conversation_history': []
    }

    session_id = str(uuid.uuid4())

    response = agent.handle_user_message(
        session_id=session_id,
        user_message=user_question,
        conversation_context=conversation_context,
        additional_context={
            "is_follow_up": False,
            "message_count": 1
        }
    )

    return response['response']

if __name__ == "__main__":
    print("🀆 TCM智能对话代理测试 🀆\n")

    # 测试对话功能
    test_questions = [
        "我最近总是感觉口干舌燥，需要吃什么中药调理？",
        "这种面色饮食上有什么需要特别注意的吗？",
        "大概需要调理多长时间才能看到改善？",
        "平时生活和运动上要注意什么？"
    ]

    for question in test_questions:
        print(f"🗣️ 用户：{question}")
        response = quick_chat_diagnosis(0, question, {"name": "李女士", "age": 35})
        print(f"🀆 AI：{response}")
        print("-" * 50)