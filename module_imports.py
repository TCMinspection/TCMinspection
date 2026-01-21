#!/usr/bin/env python3
"""
中医AI智能望诊系统 - 模块导入管理器
基于Context7最佳实践的Python模块导入解决方案
"""

import os
import sys
import importlib
from pathlib import Path

class TCMModuleImporter:
    """TCM系统模块导入管理器"""

    def __init__(self, base_dir=None):
        """
        初始化导入管理器

        Args:
            base_dir: 项目根目录，默认为当前文件所在目录
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent
        else:
            self.base_dir = Path(base_dir)

        self.module_paths = {
            'code.part1': self.base_dir / 'code.part1',
            'code.part2': self.base_dir / 'code.part2',
            'code.part4': self.base_dir / 'code.part4'
        }

        self.imported_modules = {}

    def setup_sys_path(self):
        """设置sys.path，确保Python能找到我们的模块"""
        print("🔧 设置模块搜索路径...")

        # 获取当前sys.path的快照
        original_path = sys.path.copy()

        # 将项目模块目录添加到sys.path前面（优先级最高）
        for module_name, module_path in self.module_paths.items():
            if module_path.exists():
                str_path = str(module_path)
                # 只在路径不存在时添加，避免重复
                if str_path not in sys.path:
                    sys.path.insert(0, str_path)
                    print(f"  ✅ 添加路径: {module_name} -> {str_path}")
                else:
                    print(f"  ⚠️  路径已存在: {module_name}")
            else:
                print(f"  ❌ 路径不存在: {module_path}")

        return sys.path != original_path

    def create_init_files(self):
        """创建必要的__init__.py文件"""
        print("📝 创建__init__.py文件...")

        for module_name, module_path in self.module_paths.items():
            if module_path.exists():
                init_file = module_path / '__init__.py'
                if not init_file.exists():
                    init_file.write_text(f'"""\n{module_name} 模块包\n中医AI智能望诊系统\n"""\n', encoding='utf-8')
                    print(f"  ✅ 创建: {init_file}")
                else:
                    print(f"  ⚠️  已存在: {init_file}")

    def validate_module_files(self):
        """验证关键模块文件是否存在"""
        print("📋 验证模块文件...")

        required_files = {
            'code.part1': [
                'segmentation_mindspore.py',
                'Autoencoder_mindspore.py'
            ],
            'code.part4': [
                'agent.py',
                'enhanced_agent.py'
            ]
        }

        all_exist = True
        for module_name, files in required_files.items():
            module_path = self.module_paths[module_name]
            print(f"  📁 {module_name}:")

            for file_name in files:
                file_path = module_path / file_name
                if file_path.exists():
                    print(f"    ✅ {file_name}")
                else:
                    print(f"    ❌ {file_name} - 文件不存在!")
                    all_exist = False

        return all_exist

    def test_import_module(self, module_name, import_items=None):
        """
        测试模块导入

        Args:
            module_name: 模块名称
            import_items: 要导入的项目列表，None表示导入整个模块

        Returns:
            (success, module_or_error)
        """
        try:
            if import_items is None:
                # 导入整个模块
                module = importlib.import_module(module_name)
                return True, module
            else:
                # 导入特定项目
                module = importlib.import_module(module_name)
                imported_items = []
                for item_name in import_items:
                    item = getattr(module, item_name, None)
                    if item is None:
                        return False, f"模块 {module_name} 中没有找到 {item_name}"
                    imported_items.append(item)
                return True, imported_items

        except ImportError as e:
            return False, f"导入失败: {str(e)}"
        except Exception as e:
            return False, f"其他错误: {str(e)}"

    def import_all_modules(self):
        """导入所有必需的模块"""
        print("📦 导入TCM系统模块...")

        # 定义要导入的模块和项目
        import_config = {
            'segmentation_mindspore': [
                'VGG16Features',
                'create_data_transforms1',
                'extract_features2',
                'compress_features_with_autoencoder'
            ],
            'Autoencoder_mindspore': ['SupervisedAutoencoder'],
            'agent': ['TCMComplexionAnalysisAgent'],
            'enhanced_agent': ['quick_chat_diagnosis']
        }

        successful_imports = {}
        failed_imports = {}

        for module_name, import_items in import_config.items():
            print(f"  🔄 导入 {module_name}...")
            success, result = self.test_import_module(module_name, import_items)

            if success:
                successful_imports[module_name] = result
                if import_items:
                    for i, item_name in enumerate(import_items):
                        self.imported_modules[item_name] = result[i]
                else:
                    self.imported_modules[module_name] = result
                print(f"    ✅ {module_name} 导入成功")
            else:
                failed_imports[module_name] = result
                print(f"    ❌ {module_name} 导入失败: {result}")

        return successful_imports, failed_imports

    def get_imported_module(self, name):
        """获取已导入的模块或函数"""
        return self.imported_modules.get(name)

    def run_full_setup(self):
        """运行完整的模块设置流程"""
        print("🚀 TCM系统模块导入设置")
        print("=" * 50)

        # 1. 设置sys.path
        path_changed = self.setup_sys_path()

        # 2. 创建__init__.py文件
        self.create_init_files()

        # 3. 验证文件存在性
        files_valid = self.validate_module_files()

        # 4. 导入所有模块
        successful, failed = self.import_all_modules()

        # 5. 汇总结果
        print("\n" + "=" * 50)
        print("📊 导入设置结果汇总")
        print("=" * 50)

        print(f"路径设置: {'✅ 成功' if path_changed else '⚠️  无变化'}")
        print(f"文件验证: {'✅ 全部存在' if files_valid else '❌ 有缺失文件'}")
        print(f"模块导入: ✅ {len(successful)} 成功, ❌ {len(failed)} 失败")

        if failed:
            print("\n❌ 导入失败的模块:")
            for module_name, error in failed.items():
                print(f"  - {module_name}: {error}")

        if successful:
            print("\n✅ 成功导入的模块:")
            for module_name in successful.keys():
                print(f"  - {module_name}")

        all_success = files_valid and len(failed) == 0

        if all_success:
            print("\n🎉 所有模块导入设置成功！系统准备就绪。")
            return True
        else:
            print("\n⚠️  部分模块导入失败，请检查上述错误信息。")
            return False

# 全局导入管理器实例
_tcm_importer = None

def get_tcm_importer():
    """获取全局TCM导入管理器实例"""
    global _tcm_importer
    if _tcm_importer is None:
        _tcm_importer = TCMModuleImporter()
    return _tcm_importer

def setup_tcm_modules():
    """设置TCM模块的快捷函数"""
    importer = get_tcm_importer()
    return importer.run_full_setup()

def import_tcm_modules():
    """导入TCM模块的快捷函数"""
    importer = get_tcm_importer()
    successful, failed = importer.import_all_modules()
    return successful, failed, importer.imported_modules

# 使用示例
if __name__ == "__main__":
    # 直接运行此文件进行模块设置和测试
    success = setup_tcm_modules()

    if success:
        print("\n🧪 测试模块使用...")
        importer = get_tcm_importer()

        # 测试使用导入的模块
        VGG16Features = importer.get_imported_module('VGG16Features')
        if VGG16Features:
            print("✅ VGG16Features 可以正常使用")

        TCMComplexionAnalysisAgent = importer.get_imported_module('TCMComplexionAnalysisAgent')
        if TCMComplexionAnalysisAgent:
            print("✅ TCMComplexionAnalysisAgent 可以正常使用")

        print("\n✨ 模块导入测试完成！")
    else:
        print("\n❌ 模块导入设置失败，请修复上述问题后重试。")