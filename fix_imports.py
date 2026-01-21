#!/usr/bin/env python3
"""
修复导入问题的脚本
检查并修复模块导入路径
"""

import os
import sys

def check_module_structure():
    """检查模块结构"""
    print("🔍 检查项目模块结构...")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    required_dirs = ['code.part1', 'code.part2', 'code.part4']
    for dir_name in required_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"✅ {dir_name}")
        else:
            print(f"❌ {dir_name} - 目录不存在")

    # 检查关键文件
    key_files = {
        'code.part1/segmentation_mindspore.py': 'MindSpore图像分割模块',
        'code.part1/Autoencoder_mindspore.py': 'MindSpore自编码器',
        'code.part4/enhanced_agent.py': '增强AI代理',
        'code.part4/agent.py': '基础AI代理',
        'code.part2/SVM未修改版.py': 'SVM分类器'
    }

    for file_path, description in key_files.items():
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"✅ {description}: {file_path}")
        else:
            print(f"❌ {description}: {file_path}")

def test_imports():
    """测试导入"""
    print("\n🧪 测试模块导入...")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 添加路径
    sys.path.insert(0, os.path.join(base_dir, 'code.part1'))
    sys.path.insert(0, os.path.join(base_dir, 'code.part4'))

    try:
        # 测试MindSpore模块
        print("测试MindSpore模块...")
        from segmentation_mindspore import VGG16Features, create_data_transforms1, extract_features2, compress_features_with_autoencoder
        print("✅ segmentation_mindspore 导入成功")

        from Autoencoder_mindspore import SupervisedAutoencoder
        print("✅ Autoencoder_mindspore 导入成功")

        # 测试AI模块
        print("测试AI模块...")
        from agent import TCMComplexionAnalysisAgent
        print("✅ agent 导入成功")

        from enhanced_agent import quick_chat_diagnosis
        print("✅ enhanced_agent 导入成功")

        return True

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def create_init_files():
    """创建__init__.py文件"""
    print("\n📝 创建__init__.py文件...")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    init_files = [
        'code.part1/__init__.py',
        'code.part2/__init__.py',
        'code.part4/__init__.py'
    ]

    for init_file in init_files:
        file_path = os.path.join(base_dir, init_file)
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f'"""\n{os.path.basename(os.path.dirname(file_path))} 模块\n"""\n')
            print(f"✅ 创建 {init_file}")
        else:
            print(f"✅ {init_file} 已存在")

def main():
    """主函数"""
    print("🔧 中医AI智能望诊系统 - 导入修复工具")
    print("=" * 50)

    # 检查模块结构
    check_module_structure()

    # 创建__init__.py文件
    create_init_files()

    # 测试导入
    if test_imports():
        print("\n🎉 导入测试成功！")
        print("现在可以正常运行 python app.py")
    else:
        print("\n❌ 导入测试失败")
        print("请检查模块文件是否存在且路径正确")

if __name__ == "__main__":
    main()