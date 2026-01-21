#!/usr/bin/env python3
"""
中医AI智能望诊系统 - Web API服务
集成图像分析、SVM分类、LLM报告生成和对话功能
"""

import os
import sys
import json
import uuid
import base64
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import dlib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import mindspore as ms
from mindspore import Tensor, context
from mindspore.common import dtype as mstype
from PIL import Image

# 导入自定义模块 - 使用新的导入管理器
try:
    # 导入我们的模块导入管理器
    from module_imports import setup_tcm_modules, get_tcm_importer

    print("🔧 初始化TCM模块导入系统...")

    # 设置所有模块
    setup_success = setup_tcm_modules()

    if not setup_success:
        print("❌ 模块导入设置失败，尝试备用方案...")
        raise ImportError("模块导入设置失败")

    # 获取导入管理器
    importer = get_tcm_importer()

    # 从导入管理器获取所需的模块和函数
    VGG16Features = importer.get_imported_module('VGG16Features')
    create_data_transforms1 = importer.get_imported_module('create_data_transforms1')
    extract_features2 = importer.get_imported_module('extract_features2')
    compress_features_with_autoencoder = importer.get_imported_module('compress_features_with_autoencoder')
    SupervisedAutoencoder = importer.get_imported_module('SupervisedAutoencoder')
    quick_chat_diagnosis = importer.get_imported_module('quick_chat_diagnosis')
    TCMComplexionAnalysisAgent = importer.get_imported_module('TCMComplexionAnalysisAgent')

    print("✅ 所有TCM模块导入成功")

except ImportError as e:
    print(f"❌ TCM模块导入失败: {e}")
    print("正在尝试传统导入方案...")

    # 备用传统导入方案
    try:
        # 添加项目路径到sys.path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(base_dir, 'code.part1'))
        sys.path.insert(0, os.path.join(base_dir, 'code.part4'))

        from segmentation_mindspore import (
            VGG16Features, create_data_transforms1, extract_features2,
            compress_features_with_autoencoder
        )
        from Autoencoder_mindspore import SupervisedAutoencoder
        from enhanced_agent import quick_chat_diagnosis
        from agent import TCMComplexionAnalysisAgent
        print("✅ 传统导入方案成功")

    except ImportError as e2:
        print(f"❌ 所有导入方案都失败了: {e2}")
        print("\n🔧 故障排除建议:")
        print("1. 运行 'python module_imports.py' 进行详细诊断")
        print("2. 运行 'python fix_imports.py' 尝试自动修复")
        print("3. 检查以下文件是否存在:")
        print("   - code.part1/segmentation_mindspore.py")
        print("   - code.part1/Autoencoder_mindspore.py")
        print("   - code.part4/agent.py")
        print("   - code.part4/enhanced_agent.py")
        sys.exit(1)

# 应用配置
app = Flask(__name__)
CORS(app, origins=["*"])

# 配置参数
UPLOAD_FOLDER = 'uploads'
TEMP_FOLDER = 'temp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# TCM面色类型定义
TCM_COMPLEXION_TYPES = {
    0: {"name": "红色", "icon": "🔴", "tcm_type": "阴虚火旺", "description": "面部红赤，多为阴虚内热"},
    1: {"name": "黑色", "icon": "⚫", "tcm_type": "肾虚寒盛", "description": "面色晦暗，多为肾阳不足"},
    2: {"name": "白色", "icon": "⚪", "tcm_type": "气血两虚", "description": "面色苍白，多为气血亏虚"},
    3: {"name": "黄色", "icon": "🟡", "tcm_type": "脾虚湿盛", "description": "面色萎黄，多为脾虚湿困"},
    4: {"name": "青色", "icon": "🔵", "tcm_type": "寒凝血瘀", "description": "面色青紫，多为寒凝血瘀"}
}

class TCMAnalyzer:
    """中医面色分析器 - 整合所有功能模块"""

    def __init__(self):
        """初始化分析器"""
        self.vgg_model = None
        self.autoencoder_model = None
        self.svm_pipeline = None
        self.tcm_agent = None
        self.detector = None
        self.predictor = None
        self.data_transforms = None

        # 初始化MindSpore环境
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

        print("正在初始化TCM智能分析器...")
        self._load_models()
        print("初始化完成！")

    def _load_models(self):
        """加载所有需要的模型"""
        try:
            # 1. 加载VGG16特征提取模型
            print("加载VGG16模型...")
            self.vgg_model = VGG16Features()
            self.vgg_model.set_train(False)

            # 2. 加载自编码器模型
            print("加载自编码器模型...")
            self.autoencoder_model = SupervisedAutoencoder(num_classes=5)
            autoencoder_ckpt = 'D:/TCMinspection/autoencoder_model_mindspore.ckpt'
            if os.path.exists(autoencoder_ckpt):
                param_dict = ms.load_checkpoint(autoencoder_ckpt)
                ms.load_param_into_net(self.autoencoder_model, param_dict)
                self.autoencoder_model.set_train(False)

            # 3. 加载SVM分类器
            print("加载SVM分类器...")
            import joblib
            svm_path = 'D:/TCMinspection/svm_pipeline.pkl'
            if os.path.exists(svm_path):
                self.svm_pipeline = joblib.load(svm_path)

            # 4. 加载TCM分析代理
            print("加载TCM分析代理...")
            self.tcm_agent = TCMComplexionAnalysisAgent()

            # 5. 加载面部检测器
            print("加载面部检测器...")
            self.detector = dlib.get_frontal_face_detector()
            predictor_path = "D:/dlib-predictor/shape_predictor_68_face_landmarks.dat"
            if os.path.exists(predictor_path):
                self.predictor = dlib.shape_predictor(predictor_path)

            # 6. 创建数据预处理管道
            self.data_transforms = create_data_transforms1()

            print("所有模型加载完成！")

        except Exception as e:
            print(f"模型加载失败: {e}")
            traceback.print_exc()

    def analyze_face_image(self, image_path: str) -> Dict[str, Any]:
        """
        分析面部图像，返回面色分类结果

        Args:
            image_path: 图像文件路径

        Returns:
            包含分类结果和置信度的字典
        """
        try:
            if not all([self.vgg_model, self.svm_pipeline, self.detector, self.predictor]):
                raise Exception("模型未完全加载")

            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("无法读取图像文件")

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 检测面部
            faces = self.detector(gray)
            if len(faces) == 0:
                raise Exception("未检测到面部")

            face = faces[0]
            landmarks = self.predictor(gray, face)

            # 提取特征区域（对应训练时的9个区域）
            face_regions = []
            ting_start = (landmarks.part(21).x, landmarks.part(21).y)
            ting_end = (landmarks.part(22).x, landmarks.part(22).y)
            mouth_left = (landmarks.part(48).x, landmarks.part(48).y)
            mouth_right = (landmarks.part(54).x, landmarks.part(54).y)
            chin_tip = (landmarks.part(67).x, landmarks.part(67).y)

            # 提取9个区域的特征
            region_features1 = []  # CNN特征
            region_features2 = []  # 手工特征

            # 定义9个区域的坐标
            regions = [
                (mouth_left[0]-10, mouth_left[1]-10, mouth_right[0]-mouth_left[0]+25, mouth_right[1]-mouth_left[1]+25),  # 口角
                (ting_start[0], ting_start[1]-10, ting_end[0]-ting_start[0], ting_end[1]-ting_start[1]+20),  # 阙
                (landmarks.part(19).x+5, landmarks.part(19).y-20, 95, 10),
                (landmarks.part(1).x+5, landmarks.part(1).y-10, 20, 20),
                (landmarks.part(15).x-25, landmarks.part(15).y-10, 20, 20),
                (landmarks.part(1).x+25, landmarks.part(1).y+10, 25, 30),
                (landmarks.part(15).x-50, landmarks.part(15).y+10, 25, 30),
                (landmarks.part(67).x-5, landmarks.part(67).y+20, 30, 20),
                (landmarks.part(30).x-10, landmarks.part(30).y-10, 20, 20)
            ]

            for i, (x, y, w, h) in enumerate(regions):
                # 确保坐标有效
                x, y = max(0, x), max(0, y)
                x_end = min(image.shape[1], x + w)
                y_end = min(image.shape[0], y + h)

                if x_end <= x or y_end <= y:
                    continue

                region_image = image[y:y_end, x:x_end]
                if region_image.size == 0:
                    continue

                # 提取CNN特征
                try:
                    features1 = self._extract_cnn_features(region_image)
                    if features1 is not None:
                        compressed_features = compress_features_with_autoencoder(
                            features1, self.autoencoder_model
                        )
                        compressed_features = np.asarray(compressed_features).reshape(-1)
                        region_features1.append(compressed_features)
                except Exception as e:
                    print(f"区域{i} CNN特征提取失败: {e}")
                    continue

                # 提取手工特征
                try:
                    features2 = extract_features2(region_image)
                    region_features2.append(features2)
                except Exception as e:
                    print(f"区域{i}手工特征提取失败: {e}")
                    continue

            if not region_features1 or not region_features2:
                raise Exception("特征提取失败")

            # 融合特征
            final_features = np.hstack(region_features1 + region_features2)

            # SVM分类
            predicted_type = self.svm_pipeline.predict([final_features])[0]

            # 处理字符串/数值标签的映射
            # SVM可能返回字符串标签（'white', 'black', 'yellow', 'red', 'cyan'）
            # 也可能返回数值标签（0, 1, 2, 3, 4）
            type_mapping = {
                'white': 2, 'black': 1, 'yellow': 3, 'red': 0, 'cyan': 4,
                0: 0, 1: 1, 2: 2, 3: 3, 4: 4
            }

            if predicted_type in type_mapping:
                complexion_type = type_mapping[predicted_type]
            else:
                # 如果标签不在映射中，尝试转换为整数
                try:
                    complexion_type = int(predicted_type)
                except (ValueError, TypeError):
                    print(f"未知的预测标签: {predicted_type}")
                    complexion_type = 2  # 默认为白色（气血两虚）

            # 获取分类概率
            confidence = 0.85  # 默认置信度
            try:
                if hasattr(self.svm_pipeline.named_steps['svm'], 'decision_function'):
                    decision_scores = self.svm_pipeline.decision_function([final_features])[0]
                    if len(decision_scores) > 1:
                        exp_scores = np.exp(decision_scores - np.max(decision_scores))
                        probabilities = exp_scores / np.sum(exp_scores)
                        confidence = float(probabilities[complexion_type])
            except:
                pass

            return {
                "complexion_type": int(complexion_type),
                "confidence": float(np.clip(confidence, 0.0, 1.0)),
                "complexion_info": TCM_COMPLEXION_TYPES.get(int(complexion_type), {}),
                "predicted_raw": str(predicted_type),  # 调试信息
                "status": "success"
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": f"图像分析失败: {str(e)}"
            }

    def _extract_cnn_features(self, region_image):
        """提取CNN特征"""
        try:
            # 预处理图像
            image_rgb = cv2.cvtColor(region_image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            image_array = np.array(image_pil)

            # 应用变换
            import mindspore.dataset.vision as vision
            import mindspore.dataset.transforms as transforms
            data_transforms = transforms.Compose(self.data_transforms)
            transformed = data_transforms(image_array)

            # 转换为MindSpore张量
            image_tensor = Tensor(transformed, dtype=mstype.float32)
            image_tensor = ms.ops.expand_dims(image_tensor, 0)

            # 提取特征
            self.vgg_model.set_train(False)
            features = self.vgg_model(image_tensor)
            features = ms.ops.flatten(features)

            return features.asnumpy()

        except Exception as e:
            print(f"CNN特征提取失败: {e}")
            return None

# 全局分析器实例
tcm_analyzer = None

def get_analyzer():
    """获取分析器实例（单例模式）"""
    global tcm_analyzer
    if tcm_analyzer is None:
        tcm_analyzer = TCMAnalyzer()
    return tcm_analyzer

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# API路由定义

@app.route('/api/health', methods=['GET'])
def health_check():
    """系统健康检查"""
    try:
        analyzer = get_analyzer()
        models_status = {
            "vgg_model": analyzer.vgg_model is not None,
            "autoencoder_model": analyzer.autoencoder_model is not None,
            "svm_pipeline": analyzer.svm_pipeline is not None,
            "tcm_agent": analyzer.tcm_agent is not None,
            "face_detector": analyzer.detector is not None,
            "landmark_predictor": analyzer.predictor is not None
        }

        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models": models_status,
            "message": "系统运行正常"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """
    上传并分析面部图像
    POST Data:
    - file: 图片文件
    - patient_info: 患者信息JSON（可选）
    """
    try:
        # 检查文件
        if 'file' not in request.files:
            return jsonify({"error": "未上传文件"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "未选择文件"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "不支持的文件格式"}), 400

        # 保存临时文件
        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        temp_path = os.path.join(TEMP_FOLDER, filename)
        file.save(temp_path)

        # 获取患者信息
        patient_info = {}
        if 'patient_info' in request.form:
            try:
                patient_info = json.loads(request.form['patient_info'])
            except:
                patient_info = {}

        # 分析图像
        analyzer = get_analyzer()
        result = analyzer.analyze_face_image(temp_path)

        # 添加患者信息到结果
        result['patient_info'] = patient_info
        result['timestamp'] = datetime.now().isoformat()
        result['image_id'] = filename

        # 清理临时文件
        try:
            os.remove(temp_path)
        except:
            pass

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "分析失败",
            "message": str(e)
        }), 500

@app.route('/api/report', methods=['POST'])
def generate_report():
    """
    生成详细检测报告
    POST Data:
    - complexion_type: 面色类型 (0-4)
    - confidence: 置信度
    - patient_info: 患者信息
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "请求数据为空"}), 400

        complexion_type = data.get('complexion_type')
        confidence = data.get('confidence', 0.0)
        patient_info = data.get('patient_info', {})

        if complexion_type is None:
            return jsonify({"error": "缺少面色类型参数"}), 400

        # 生成报告
        analyzer = get_analyzer()
        report = analyzer.tcm_agent.quick_analysis(
            complexion_type=complexion_type,
            confidence=confidence,
            patient_info=patient_info
        )

        return jsonify({
            "report": report,
            "complexion_type": complexion_type,
            "complexion_info": TCM_COMPLEXION_TYPES.get(int(complexion_type), {}),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "报告生成失败",
            "message": str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    """
    与AI进行健康咨询对话
    POST Data:
    - complexion_type: 面色类型 (0-4)
    - message: 用户消息
    - patient_info: 患者信息
    - session_id: 会话ID（可选）
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "请求数据为空"}), 400

        complexion_type = data.get('complexion_type')
        message = data.get('message', '').strip()
        patient_info = data.get('patient_info', {})
        session_id = data.get('session_id', str(uuid.uuid4()))

        if complexion_type is None:
            return jsonify({"error": "缺少面色类型参数"}), 400

        if not message:
            return jsonify({"error": "消息不能为空"}), 400

        # 确保complexion_type是整数类型
        try:
            complexion_type = int(complexion_type)
        except (ValueError, TypeError):
            complexion_type = 2  # 默认为白色（气血两虚）

        # 生成AI回复
        response = quick_chat_diagnosis(
            complexion_type=complexion_type,
            user_question=message,
            patient_info=patient_info
        )

        return jsonify({
            "response": response,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "对话失败",
            "message": str(e)
        }), 500

@app.route('/api/complexion_types', methods=['GET'])
def get_complexion_types():
    """获取所有面色类型信息"""
    return jsonify({
        "complexion_types": TCM_COMPLEXION_TYPES,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """访问上传的文件"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.errorhandler(413)
def too_large(e):
    """文件过大错误处理"""
    return jsonify({"error": "文件过大，请上传小于16MB的图片"}), 413

@app.errorhandler(404)
def not_found(e):
    """404错误处理"""
    return jsonify({"error": "接口不存在"}), 404

@app.route('/')
def intro():
    """中医知识介绍页面（首页）"""
    return send_from_directory('templates', 'intro.html')

@app.route('/analyze')
def index():
    """望诊系统页面"""
    return send_from_directory('templates', 'index.html')

@app.route('/test_api.html')
def test_api():
    """API测试页面"""
    return send_from_directory('.', 'test_api.html')

@app.errorhandler(500)
def internal_error(e):
    """500错误处理"""
    return jsonify({"error": "服务器内部错误"}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("🏥 中医AI智能望诊系统")
    print("=" * 50)
    print("📊 功能模块:")
    print("  ✅ 面部图像分析")
    print("  ✅ SVM面色分类")
    print("  ✅ TCM智能报告生成")
    print("  ✅ AI健康咨询对话")
    print("=" * 50)
    print("🌐 API服务地址: http://localhost:5000")
    print("📖 API文档:")
    print("  POST /api/analyze - 图像分析")
    print("  POST /api/report - 生成报告")
    print("  POST /api/chat - AI对话")
    print("  GET  /api/health - 系统状态")
    print("  GET  /api/complexion_types - 面色类型")
    print("=" * 50)

    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=True)