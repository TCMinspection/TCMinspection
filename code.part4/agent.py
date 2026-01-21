from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
import os
import datetime
import json
from typing import Dict, Any, Optional, List

# 设置 OpenRouter API Key
os.environ["OPENAI_API_KEY"] = "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"  # 注意：出于安全考虑，建议从环境变量读取

# TCM 面色分类定义
TCM_COMPLEXION_TYPES = {
    0: "红色",
    1: "黑色",
    2: "白色",
    3: "黄色",
    4: "青色"
}

class TCMComplexionAnalysisAgent:
    """TCM面色分析代理 - 集成LLM和RAG功能"""

    def __init__(self,
                knowledge_urls: Optional[List[str]] = None,
                output_dir: str = "./reports",
                model_name: str = "openai/gpt-5"):
        """
        初始化TCM面色分析代理

        Args:
            knowledge_urls: 知识库URL列表
            output_dir: 报告输出目录
            model_name: LLM模型名称
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 初始化LLM
        self.llm = ChatOpenAI(
            model=model_name,
            base_url="https://openrouter.ai/api/v1"
        )

        # 初始化RAG系统
        self.knowledge_urls = knowledge_urls or [
            "https://www.scm.cuhk.edu.hk/en-gb/articles-redirect/236-hkej/1767-2020-10-02",
        ]
        self._init_rag_system()

        print("✅ TCM面色分析代理初始化完成")

    def _init_rag_system(self):
        """初始化RAG知识库系统"""
        try:
            print("🔄 初始化RAG知识库...")

            # 加载文档
            loader = WebBaseLoader(
                web_paths=self.knowledge_urls,
                header_template={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            docs = loader.load()

            # 添加元数据
            for i, doc in enumerate(docs):
                doc.metadata["source"] = doc.metadata.get("source", f"doc_{i}")

            # 文本切分
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)

            # 向量化
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
            self.vectorstore = FAISS.from_documents(splits, embeddings)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

            print("✅ RAG系统初始化成功")

        except Exception as e:
            print(f"⚠️ RAG系统初始化失败: {e}")
            self.vectorstore = None
            self.retriever = None

    def generate_detection_report(self,
                                classification_result: Dict[str, Any],
                                patient_info: Optional[Dict[str, Any]] = None,
                                include_rag_knowledge: bool = True) -> str:
        """
        生成面色检测报告（增强版，包含LLM分析）

        Args:
            classification_result: 分类结果，必须包含 'complexion_type' 字段
            patient_info: 患者信息
            include_rag_knowledge: 是否使用RAG知识增强

        Returns:
            检测报告文本
        """
        complexion_type = classification_result.get('complexion_type', 0)
        complexion_name = TCM_COMPLEXION_TYPES.get(complexion_type, "未知类型")
        confidence = classification_result.get('confidence')

        # 基础报告信息
        report_time = datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

        if patient_info is None:
            patient_info = {"姓名": "匿名", "年龄": "未知", "性别": "未知"}

        # 构建基础报告框架
        basic_report = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                    TCM面色检测报告                    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

【基本信息】
检测时间: {report_time}
患者信息:
    - 姓名: {patient_info.get('姓名', '匿名')}
    - 年龄: {patient_info.get('年龄', '未知')}
    - 性别: {patient_info.get('性别', '未知')}

【检测结果】
面色类型: {complexion_name} (类型编码: {complexion_type})
"""

        if confidence is not None:
            confidence_level = "高" if confidence > 0.8 else "中" if confidence > 0.6 else "低"
            basic_report += f"置信度: {confidence:.2%} (置信等级: {confidence_level})\n"

        # 使用LLM生成增强分析
        enhanced_analysis = self._generate_llm_analysis(
            complexion_type, complexion_name, confidence, include_rag_knowledge
        )

        # 合并生成完整报告
        full_report = basic_report + "\n【智能分析】\n" + enhanced_analysis + "\n"

        full_report += f"""
【注意事项】
• 本报告由AI辅助生成，仅供参考，不能替代专业医疗诊断
• 如有身体不适，请及时就医咨询专业中医师
• 建议结合舌诊、脉诊等其他诊断方法综合判断

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            检测系统: TCM面色智能分析仪 v2.0
            报告生成时间: {report_time}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        return full_report.strip()

    def _generate_llm_analysis(self,
                            complexion_type: int,
                            complexion_name: str,
                            confidence: Optional[float],
                            include_rag_knowledge: bool) -> str:
        """使用LLM生成增强分析"""

        conf_str = f"{confidence:.2%}" if confidence else "未提供"

        base_prompt = f"""
        您是TCM（中医）专家，请基于以下信息提供专业的面色分析：

        面色类型: {complexion_name} (编码: {complexion_type})
        置信度: {conf_str}

        请从以下几个方面进行分析：
        1. TCM理论解释：这种面色的病理机制和临床意义
        2. 病因病机：可能导致这种面色的原因
        3. 辨证要点：如何与其他相似证候鉴别
        4. 调理原则：总体的调理思路
        5. 生活建议：饮食、作息、情绪等具体建议
        6. 预警信号：需要特别注意的症状变化

        要求：
        - 分析要专业准确，但表述要通俗易懂
        - 建议要具体可操作
        - 体现中医整体观念和辨证论治思想
        """


        if include_rag_knowledge and self.retriever is not None:
            # RAG增强版本
            rag_chain = (
                {
                    "context": self.retriever | self._format_docs,
                    "question": lambda _: base_prompt
                }
                | ChatPromptTemplate.from_template("""
                基于以下知识库信息，请提供更全面的TCM面色分析：

                知识库内容:
                {context}

                用户问题:
                {question}

                请提供详细的专业分析，并在适当的地方引用相关知识。
                """)
                | self.llm
                | StrOutputParser()
            )

            try:
                analysis = rag_chain.invoke("TCM面色分析")
                return analysis
            except Exception as e:
                print(f"RAG分析失败，使用基础分析: {e}")
                return self._basic_llm_analysis(base_prompt)
        else:
            return self._basic_llm_analysis(base_prompt)

    def _basic_llm_analysis(self, prompt: str) -> str:
        """基础LLM分析（不使用RAG）"""
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"LLM分析失败: {e}")
            return self._fallback_analysis()

    def _fallback_analysis(self) -> str:
        """回退到本地知识库分析"""
        # 这里是简化的回退逻辑
        fallback_knowledge = {
            0: "红色面色多提示体内有热，需清热泻火。注意避免辛辣刺激食物。",
            1: "黑色面色多提示肾虚寒盛，需温补肾阳。注意保暖避寒。",
            2: "白色面色多提示气血亏虚，需补益气血。注意休息调养。",
            3: "黄色面色多提示脾虚湿盛，需健脾祛湿。注意饮食调理。",
            4: "青色面色多提示寒凝血瘀，需温经散寒。注意运动活血。"
        }

        return "基于本地知识库的分析:\n" + fallback_knowledge.get(0, "暂无具体分析数据")

    def _format_docs(self, docs) -> str:
        """格式化文档内容"""
        return "\n\n".join([f"[{i+1}] {doc.page_content}\n(来源: {doc.metadata.get('source', '未知')})"
                        for i, doc in enumerate(docs)])

    def save_report(self, report_text: str, filename: Optional[str] = None) -> str:
        """保存报告到文件"""
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tcm_complexion_report_{timestamp}.txt"

        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"✅ 报告已保存: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ 保存报告失败: {e}")
            return None

    def quick_analysis(self,
                    complexion_type: int,
                    confidence: Optional[float] = None,
                    patient_info: Optional[Dict[str, Any]] = None,
                    save_report: bool = True) -> str:
        """
        快速面色分析（简化接口）

        Args:
            complexion_type: 面色类型 0-4
            confidence: 置信度
            patient_info: 患者信息
            save_report: 是否保存报告

        Returns:
            分析报告文本
        """
        classification_result = {
            'complexion_type': complexion_type,
            'confidence': confidence
        }

        report = self.generate_detection_report(
            classification_result,
            patient_info,
            include_rag_knowledge=True
        )

        if save_report:
            self.save_report(report)

        return report

def quick_report(complexion_type: int,
                confidence: Optional[float] = None,
                patient_info: Optional[Dict[str, Any]] = None,
                save_to_file: bool = True,
                enable_rag: bool = True) -> str:
    """
    快捷函数：快速生成面色检测报告

    Args:
        complexion_type: 面色类型 (0-4)
        confidence: 置信度 (可选)
        patient_info: 患者信息 (可选)
        save_to_file: 是否保存到文件
        enable_rag: 是否启用RAG功能

    Returns:
        检测报告文本
    """
    agent = TCMComplexionAnalysisAgent(knowledge_urls=None if not enable_rag else None)
    return agent.quick_analysis(complexion_type, confidence, patient_info, save_to_file)

# 添加缺失的导入
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    print("⚠️ 找不到HuggingFaceEmbeddings，RAG功能将受限")
    HuggingFaceEmbeddings = None

if __name__ == "__main__":
    print("🀆 TCM面色智能分析代理 🀆\n")

    # 创建分析代理
    agent = TCMComplexionAnalysisAgent()

    # 演示：快速分析
    print("=== 演示: 智能面色分析 ===")
    report = agent.quick_analysis(
        complexion_type=3,  # 黄色
        confidence=0.87,
        patient_info={"姓名": "张三", "年龄": "35岁", "性别": "男"}
    )

    print(report)
    print("\n" + "="*60 + "\n")

    print(f"💡 使用方法:")
    print(f"1. 快速分析: quick_report(complexion_type, confidence, patient_info)")
    print(f"2. 详细分析: TCMComplexionAnalysisAgent().generate_detection_report(result)")
    print(f"3. 面色类型: {TCM_COMPLEXION_TYPES}")
    print(f"\n📁 报告保存目录: {agent.output_dir}")