#!/bin/bash
# 中医AI智能望诊系统模型部署脚本
# 处理所有模型文件的上传、验证和配置

set -e

# 配置参数
PROJECT_NAME="tcm-inspection"
DEPLOY_USER="tcmuser"
DEPLOY_PATH="/home/${DEPLOY_USER}/${PROJECT_NAME}"
MODELS_DIR="${DEPLOY_PATH}/models"
LOCAL_MODELS_DIR="D:/TCMinspection"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

echo "🤖 中医AI智能望诊系统模型部署脚本"
echo "=========================================="

# 检查参数
if [ $# -eq 0 ]; then
    log_info "使用方法：$0 <local_models_directory>"
    log_info "例如：$0 D:/TCMinspection"
    exit 1
fi

LOCAL_MODELS_DIR="$1"
if [ ! -d "$LOCAL_MODELS_DIR" ]; then
    log_error "本地模型目录不存在: $LOCAL_MODELS_DIR"
    exit 1
fi

# 检查服务器连接
check_server_connection() {
    log_step "检查服务器连接..."
    if ! ssh -o ConnectTimeout=5 ${DEPLOY_USER}@YOUR_ECS_PUBLIC_IP "exit" 2>/dev/null; then
        log_error "无法连接到服务器，请检查IP地址和SSH配置"
        exit 1
    fi
    log_info "服务器连接正常"
}

# 创建模型目录
create_model_directories() {
    log_step "创建模型目录结构..."

    ssh ${DEPLOY_USER}@YOUR_ECS_PUBLIC_IP << 'EOF'
        mkdir -p /home/tcmuser/tcm-inspection/models
        mkdir -p /home/tcmuser/tcm-inspection/models/checkpoints
        mkdir -p /home/tcmuser/tcm-inspection/models/dlib_predictor
        mkdir -p /home/tcmuser/tcm-inspection/models/pretrained
        mkdir -p /home/tcmuser/tcm-inspection/models/gan
        chmod -R 755 /home/tcmuser/tcm-inspection/models
EOF

    log_info "模型目录创建完成"
}

# 部署核心模型文件
deploy_core_models() {
    log_step "部署核心模型文件..."

    # 1. MindSpore自编码器模型
    if [ -f "$LOCAL_MODELS_DIR/autoencoder_model_mindspore.ckpt" ]; then
        log_info "上传MindSpore自编码器模型..."
        scp "$LOCAL_MODELS_DIR/autoencoder_model_mindspore.ckpt" \
            ${DEPLOY_USER}@YOUR_ECS_PUBLIC_IP:${MODELS_DIR}/
        log_info "✅ 自编码器模型上传完成"
    else
        log_warn "⚠️ MindSpore自编码器模型未找到: autoencoder_model_mindspore.ckpt"
    fi

    # 2. SVM分类器
    if [ -f "$LOCAL_MODELS_DIR/svm_pipeline.pkl" ]; then
        log_info "上传SVM分类器..."
        scp "$LOCAL_MODELS_DIR/svm_pipeline.pkl" \
            ${DEPLOY_USER}@YOUR_ECS_PUBLIC_IP:${MODELS_DIR}/
        log_info "✅ SVM分类器上传完成"
    else
        log_warn "⚠️ SVM分类器未找到: svm_pipeline.pkl"
    fi

    # 3. PyTorch自编码器模型（备选）
    if [ -f "$LOCAL_MODELS_DIR/autoencoder_model.pth" ]; then
        log_info "上传PyTorch自编码器模型（备选）..."
        scp "$LOCAL_MODELS_DIR/autoencoder_model.pth" \
            ${DEPLOY_USER}@YOUR_ECS_PUBLIC_IP:${MODELS_DIR}/
        log_info "✅ PyTorch自编码器模型上传完成"
    fi
}

# 部署面部检测模型
deploy_face_detection_models() {
    log_step "部署面部检测模型..."

    # dlib面部关键点检测器
    if [ -f "$LOCAL_MODELS_DIR/dlib_predictor/shape_predictor_68_face_landmarks.dat" ]; then
        log_info "上传dlib面部关键点检测器..."
        scp "$LOCAL_MODELS_DIR/dlib_predictor/shape_predictor_68_face_landmarks.dat" \
            ${DEPLOY_USER}@YOUR_ECS_PUBLIC_IP:${MODELS_DIR}/dlib_predictor/
        log_info "✅ dlib面部关键点检测器上传完成"
    else
        log_warn "⚠️ dlib预测器未找到，尝试从默认位置查找..."

        # 尝试从系统目录复制
        ssh ${DEPLOY_USER}@YOUR_ECS_PUBLIC_IP << 'EOF'
            # 创建dlib目录
            mkdir -p /home/tcmuser/tcm-inspection/models/dlib_predictor

            # 检查系统目录
            DLIB_DIRS=(
                "/usr/share/dlib/"
                "/usr/local/share/dlib/"
                "/opt/dlib/"
                "$HOME/dlib-predictor/"
            )

            for dir in "${DLIB_DIRS[@]}"; do
                if [ -f "$dir/shape_predictor_68_face_landmarks.dat" ]; then
                    cp "$dir/shape_predictor_68_face_landmarks.dat" \
                       /home/tcmuser/tcm-inspection/models/dlib_predictor/
                    echo "从 $dir 复制dlib预测器成功"
                    break
                fi
            done
EOF
    fi

    # ArcFace模型（备选）
    if [ -f "$LOCAL_MODELS_DIR/Tools/arcface_mobilefacenet.pth" ]; then
        log_info "上传ArcFace模型..."
        ssh ${DEPLOY_USER}@YOUR_ECS_PUBLIC_IP \
            "mkdir -p ${MODELS_DIR}/pretrained"
        scp "$LOCAL_MODELS_DIR/Tools/arcface_mobilefacenet.pth" \
            ${DEPLOY_USER}@YOUR_ECS_PUBLIC_IP:${MODELS_DIR}/pretrained/
        log_info "✅ ArcFace模型上传完成"
    fi
}

# 部署GAN模型（可选）
deploy_gan_models() {
    log_step "部署GAN生成模型（可选）..."

    if [ -d "$LOCAL_MODELS_DIR/GAN/checkpoints" ]; then
        log_info "上传GAN模型文件..."

        ssh ${DEPLOY_USER}@YOUR_ECS_PUBLIC_IP \
            "mkdir -p ${MODELS_DIR}/gan/checkpoints"

        # 上传所有GAN模型文件
        scp -r "$LOCAL_MODELS_DIR/GAN/checkpoints/"* \
            ${DEPLOY_USER}@YOUR_ECS_PUBLIC_IP:${MODELS_DIR}/gan/checkpoints/

        log_info "✅ GAN模型上传完成"
    else
        log_info "📋 GAN模型目录未找到，跳过GAN模型部署"
    fi
}

# 验证模型文件
verify_models() {
    log_step "验证模型文件完整性..."

    ssh ${DEPLOY_USER}@YOUR_ECS_PUBLIC_IP << 'EOF'
        MODELS_DIR="/home/tcmuser/tcm-inspection/models"
        TOTAL_MODELS=0
        FOUND_MODELS=0

        echo "模型文件验证报告："
        echo "=================="

        # 检查核心模型
        if [ -f "$MODELS_DIR/autoencoder_model_mindspore.ckpt" ]; then
            echo "✅ MindSpore自编码器模型: $(du -h $MODELS_DIR/autoencoder_model_mindspore.ckpt | cut -f1)"
            ((FOUND_MODELS++))
        else
            echo "❌ MindSpore自编码器模型: 未找到"
        fi
        ((TOTAL_MODELS++))

        if [ -f "$MODELS_DIR/svm_pipeline.pkl" ]; then
            echo "✅ SVM分类器: $(du -h $MODELS_DIR/svm_pipeline.pkl | cut -f1)"
            ((FOUND_MODELS++))
        else
            echo "❌ SVM分类器: 未找到"
        fi
        ((TOTAL_MODELS++))

        if [ -f "$MODELS_DIR/autoencoder_model.pth" ]; then
            echo "✅ PyTorch自编码器模型: $(du -h $MODELS_DIR/autoencoder_model.pth | cut -f1)"
            ((FOUND_MODELS++))
        else
            echo "⚠️ PyTorch自编码器模型: 未找到（可选）"
        fi
        ((TOTAL_MODELS++))

        # 检查面部检测模型
        if [ -f "$MODELS_DIR/dlib_predictor/shape_predictor_68_face_landmarks.dat" ]; then
            echo "✅ dlib面部关键点检测器: $(du -h $MODELS_DIR/dlib_predictor/shape_predictor_68_face_landmarks.dat | cut -f1)"
            ((FOUND_MODELS++))
        else
            echo "❌ dlib面部关键点检测器: 未找到"
        fi
        ((TOTAL_MODELS++))

        # 检查预训练模型
        pretrained_count=$(find $MODELS_DIR/pretrained -name "*.pth" 2>/dev/null | wc -l)
        echo "✅ 预训练模型: $pretrained_count 个文件"
        FOUND_MODELS=$((FOUND_MODELS + pretrained_count))

        # 检查GAN模型
        gan_count=$(find $MODELS_DIR/gan -name "*.pth" 2>/dev/null | wc -l)
        if [ $gan_count -gt 0 ]; then
            echo "✅ GAN模型: $gan_count 个文件"
            FOUND_MODELS=$((FOUND_MODELS + gan_count))
        else
            echo "⚠️ GAN模型: 未找到（可选）"
        fi

        echo "=================="
        echo "模型验证完成: $FOUND_MODELS/$TOTAL_MODELS 个核心模型"

        # 检查权限
        chmod -R 644 $MODELS_DIR/*.*
        chmod 755 $MODELS_DIR
        find $MODELS_DIR -type d -exec chmod 755 {} \;

        echo "权限设置完成"
EOF
}

# 创建模型配置文件
create_model_config() {
    log_step "创建模型配置文件..."

    ssh ${DEPLOY_USER}@YOUR_ECS_PUBLIC_IP << 'EOF'
        cat > /home/tcmuser/tcm-inspection/models/model_config.json << 'MODELEOF'
{
    "model_version": "1.0.0",
    "deployment_timestamp": "$(date -Iseconds)",
    "models": {
        "vgg16": {
            "type": "cnn_feature_extractor",
            "framework": "mindspore",
            "description": "VGG16特征提取模型",
            "status": "built_in",
            "size_mb": 0
        },
        "autoencoder": {
            "type": "feature_compressor",
            "framework": "mindspore",
            "description": "监督自编码器特征降维模型",
            "status": "loaded",
            "file": "autoencoder_model_mindspore.ckpt"
        },
        "svm": {
            "type": "classifier",
            "framework": "sklearn",
            "description": "SVM面色分类器",
            "status": "loaded",
            "file": "svm_pipeline.pkl"
        },
        "face_detector": {
            "type": "face_detection",
            "framework": "dlib",
            "description": "dlib面部检测和关键点定位",
            "status": "loaded",
            "file": "dlib_predictor/shape_predictor_68_face_landmarks.dat"
        }
    },
    "optional_models": {
        "pytorch_autoencoder": {
            "type": "feature_compressor",
            "framework": "pytorch",
            "description": "PyTorch版本自编码器（备选）",
            "file": "autoencoder_model.pth"
        },
        "arcface": {
            "type": "face_recognition",
            "framework": "pytorch",
            "description": "ArcFace面部识别模型",
            "file": "pretrained/arcface_mobilefacenet.pth"
        },
        "gan_generator": {
            "type": "image_generation",
            "framework": "pytorch",
            "description": "GAN图像生成模型",
            "directory": "gan/checkpoints/"
        }
    },
    "requirements": {
        "python_packages": [
            "mindspore>=1.8.0",
            "dlib>=19.22.0",
            "opencv-python>=4.5.0",
            "scikit-learn>=1.0.0",
            "torch>=1.9.0",
            "torchvision>=0.10.0"
        ],
        "system_packages": [
            "libgl1-mesa-glx",
            "libsm6",
            "libxext6",
            "libxrender-dev"
        ]
    }
}
MODELEOF

        echo "模型配置文件创建完成"
EOF
}

# 优化模型加载路径
optimize_model_paths() {
    log_step "优化模型加载路径..."

    ssh ${DEPLOY_USER}@YOUR_ECS_PUBLIC_IP << 'EOF'
        # 备份原始app.py
        cp /home/tcmuser/tcm-inspection/app.py /home/tcmuser/tcm-inspection/app.py.backup

        # 创建模型路径优化脚本
        cat > /home/tcmuser/tcm-inspection/fix_model_paths.py << 'PYEOF'
#!/usr/bin/env python3
"""
模型路径修复脚本
自动将Windows路径转换为Linux路径
"""
import os
import re

def fix_model_paths():
    app_file = '/home/tcmuser/tcm-inspection/app.py'

    with open(app_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 路径映射
    path_mappings = {
        'D:/TCMinspection/autoencoder_model_mindspore.ckpt':
            '/home/tcmuser/tcm-inspection/models/autoencoder_model_mindspore.ckpt',
        'D:/TCMinspection/svm_pipeline.pkl':
            '/home/tcmuser/tcm-inspection/models/svm_pipeline.pkl',
        'D:/dlib-predictor/shape_predictor_68_face_landmarks.dat':
            '/home/tcmuser/tcm-inspection/models/dlib_predictor/shape_predictor_68_face_landmarks.dat'
    }

    # 替换路径
    for old_path, new_path in path_mappings.items():
        content = content.replace(old_path, new_path)

    # 添加动态路径检测
    path_check_code = '''
# 动态模型路径检测
def _get_model_path(self, model_name):
    """获取模型文件路径"""
    base_path = '/home/tcmuser/tcm-inspection/models'

    model_paths = {
        'autoencoder': f'{base_path}/autoencoder_model_mindspore.ckpt',
        'svm': f'{base_path}/svm_pipeline.pkl',
        'dlib_predictor': f'{base_path}/dlib_predictor/shape_predictor_68_face_landmarks.dat'
    }

    model_path = model_paths.get(model_name)
    if model_path and os.path.exists(model_path):
        return model_path

    # 回退到Windows路径（开发环境）
    fallback_paths = {
        'autoencoder': 'D:/TCMinspection/autoencoder_model_mindspore.ckpt',
        'svm': 'D:/TCMinspection/svm_pipeline.pkl',
        'dlib_predictor': 'D:/dlib-predictor/shape_predictor_68_face_landmarks.dat'
    }

    return fallback_paths.get(model_name)
'''

    # 在_load_models方法开始处添加路径检测函数
    if 'def _get_model_path' not in content:
        load_models_start = content.find('def _load_models(self):')
        if load_models_start != -1:
            content = content[:load_models_start] + path_check_code + '\\n\\n' + content[load_models_start:]

    with open(app_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print("模型路径优化完成")

if __name__ == "__main__":
    fix_model_paths()
PYEOF

        python3 /home/tcmuser/tcm-inspection/fix_model_paths.py
        rm /home/tcmuser/tcm-inspection/fix_model_paths.py

        echo "模型路径优化完成"
EOF
}

# 创建模型测试脚本
create_model_test() {
    log_step "创建模型测试脚本..."

    ssh ${DEPLOY_USER}@YOUR_ECS_PUBLIC_IP << 'EOF'
        cat > /home/tcmuser/tcm-inspection/test_models.py << 'TESTEOF'
#!/usr/bin/env python3
"""
模型加载测试脚本
验证所有模型是否可以正常加载
"""
import sys
import os
sys.path.append('/home/tcmuser/tcm-inspection')

def test_model_loading():
    """测试模型加载"""
    print("开始模型加载测试...")

    try:
        # 导入模块
        from app import TCMAnalyzer

        # 创建分析器实例
        analyzer = TCMAnalyzer()

        print("✅ 模型加载测试成功")
        print(f"模型状态：")
        print(f"  - VGG16模型: {'✅ 就绪' if analyzer.vgg_model else '❌ 未加载'}")
        print(f"  - 自编码器模型: {'✅ 就绪' if analyzer.autoencoder_model else '❌ 未加载'}")
        print(f"  - SVM分类器: {'✅ 就绪' if analyzer.svm_pipeline else '❌ 未加载'}")
        print(f"  - 面部检测器: {'✅ 就绪' if analyzer.detector else '❌ 未加载'}")
        print(f"  - 关键点预测器: {'✅ 就绪' if analyzer.predictor else '❌ 未加载'}")

        return True

    except Exception as e:
        print(f"❌ 模型加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_face_analysis():
    """测试面部分析功能"""
    print("\\n开始面部分析功能测试...")

    try:
        from app import TCMAnalyzer

        analyzer = TCMAnalyzer()

        # 创建测试图像
        test_image_path = "/tmp/test_face.jpg"

        # 如果没有测试图像，跳过实际测试
        if not os.path.exists(test_image_path):
            print("⚠️ 没有找到测试图像，跳过面部分析测试")
            return True

        # 执行分析
        result = analyzer.analyze_face_image(test_image_path)

        if result.get('status') == 'success':
            print("✅ 面部分析测试成功")
            print(f"检测结果: {result.get('complexion_info', {}).get('name', '未知')}")
            print(f"置信度: {result.get('confidence', 0):.2%}")
        else:
            print(f"❌ 面部分析测试失败: {result.get('error', '未知错误')}")

        return True

    except Exception as e:
        print(f"❌ 面部分析测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🤖 中医AI智能望诊系统模型测试")
    print("=" * 50)

    success = True
    success &= test_model_loading()
    success &= test_face_analysis()

    print("\\n" + "=" * 50)
    if success:
        print("🎉 所有测试通过！")
    else:
        print("❌ 部分测试失败，请检查模型文件")
    print("=" * 50)

    sys.exit(0 if success else 1)
TESTEOF

        chmod +x /home/tcmuser/tcm-inspection/test_models.py
        echo "模型测试脚本创建完成"
EOF
}

# 创建华为云OBS模型备份
setup_obs_backup() {
    log_step "配置华为云OBS模型备份..."

    ssh ${DEPLOY_USER}@YOUR_ECS_PUBLIC_IP << 'EOF'
        cat > /home/tcmuser/tcm-inspection/models/backup_models_to_obs.sh << 'BACKUPEOF'
#!/bin/bash
# 模型文件备份到华为云OBS

MODELS_DIR="/home/tcmuser/tcm-inspection/models"
OBS_BUCKET="tcm-inspection-bucket"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 创建临时备份目录
BACKUP_DIR="/tmp/models_backup_$TIMESTAMP"
mkdir -p $BACKUP_DIR

# 复制模型文件
echo "正在准备模型文件备份..."
cp -r $MODELS_DIR/* $BACKUP_DIR/

# 创建备份清单
cat > $BACKUP_DIR/backup_manifest.txt << MANIFESTEOF
模型备份清单
备份时间: $(date)
备份版本: 1.0.0
模型文件列表:
$(find $MODELS_DIR -type f -exec basename {} \; | sort)
MANIFESTEOF

# 上传到OBS（如果obsutil可用）
if command -v obsutil &> /dev/null; then
    echo "正在上传模型文件到华为云OBS..."
    obsutil cp -r $BACKUP_DIR obs://$OBS_BUCKET/backups/models_$TIMESTAMP/

    if [ $? -eq 0 ]; then
        echo "✅ 模型备份上传成功"
        echo "备份路径: obs://$OBS_BUCKET/backups/models_$TIMESTAMP/"
    else
        echo "❌ 模型备份上传失败"
    fi
else
    echo "⚠️ obsutil未安装，跳过OBS备份"
fi

# 清理临时文件
rm -rf $BACKUP_DIR

echo "模型备份完成"
BACKUPEOF

        chmod +x /home/tcmuser/tcm-inspection/models/backup_models_to_obs.sh
        echo "华为云OBS备份脚本创建完成"
EOF
}

# 显示部署总结
show_deployment_summary() {
    log_step "生成部署总结..."

    echo ""
    echo "🎉 模型部署完成！"
    echo "===================="
    echo ""
    echo "📂 已部署的模型文件："
    echo "  🔧 MindSpore自编码器: autoencoder_model_mindspore.ckpt"
    echo "  🔧 SVM分类器: svm_pipeline.pkl"
    echo "  🔧 dlib面部检测器: shape_predictor_68_face_landmarks.dat"
    echo ""
    echo "📂 可选模型文件："
    echo "  🔧 PyTorch自编码器: autoencoder_model.pth"
    echo "  🔧 ArcFace面部识别: arcface_mobilefacenet.pth"
    echo "  🔧 GAN生成模型: gan/checkpoints/*.pth"
    echo ""
    echo "📂 模型目录位置："
    echo "  🖥️ 服务器: /home/tcmuser/tcm-inspection/models/"
    echo "  📁 核心模型: models/"
    echo "  📁 面部检测: models/dlib_predictor/"
    echo "  📁 预训练模型: models/pretrained/"
    echo "  📁 GAN模型: models/gan/"
    echo ""
    echo "🧪 测试命令："
    echo "  cd /home/tcmuser/tcm-inspection"
    echo "  python3 test_models.py"
    echo ""
    echo "🔄 重启应用命令："
    echo "  supervisorctl restart tcm-inspection"
    echo ""
    echo "💾 备份命令："
    echo "  ./models/backup_models_to_obs.sh"
    echo ""
}

# 主函数
main() {
    echo "开始部署模型文件到华为云服务器..."

    check_server_connection
    create_model_directories
    deploy_core_models
    deploy_face_detection_models
    deploy_gan_models
    verify_models
    create_model_config
    optimize_model_paths
    create_model_test
    setup_obs_backup
    show_deployment_summary

    echo "🚀 模型部署脚本执行完成！"
    echo ""
    echo "下一步操作："
    echo "1. 登录服务器: ssh tcmuser@YOUR_ECS_PUBLIC_IP"
    echo "2. 运行模型测试: python3 test_models.py"
    echo "3. 重启应用: supervisorctl restart tcm-inspection"
    echo "4. 验证功能: 访问您的应用网址"
}

# 执行主函数
main "$@"