import torch
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
import shutil
import json
from collections import defaultdict
import glob
import time
import random

class TCMDatasetBalancer:
    """中医面色数据集平衡器 - 使用Diffusion进行类别间转换"""

    def __init__(self, model_id="BAAI/FaceChain-Face-Portrait", target_config=None, device="cuda"):
        """
        初始化平衡器

        Args:
            model_id: 使用的diffusion模型
            target_config: 目标分布配置，例如：{"black": 50, "white": 50, "yellow": 50, "red": 50}
            device: 运行设备
        """
        self.model_id = model_id
        self.device = device
        self.target_config = target_config or {
            "black": 100,   # 目标：100张
            "white": 100,   # 目标：100张
            "yellow": 100,  # 目标：100张
            "red": 100      # 目标：100张
        }

        # 面色转换方向的prompt映射 - 针对Dataset Balancing优化版
        self.conversion_prompts = {
            # Black → 其他 (去暗增明亮)
            "black_to_yellow": [
                "{source}dark dull complexion, remove shadowy undertones → {target}warm golden healthy skin tone, natural lighting, maintain facial structure",
                "{source}grayish dark skin, reduce shadow depth → {target}sunny warm complexion, natural glow, same person",
                "{source}overly dark skin tone, lift shadow areas → {target}healthy yellow undertone, balanced appearance"
            ],

            "black_to_white": [
                "{source}dark shadowy skin, reduce darkness → {target}fair bright complexion, remove shadows, same facial features",
                "{source}deep dull complexion, illuminate skin → {target}clean white skin tone, brighten appearance naturally",
                "{source}dark undertones, increase brightness → {target}healthy pale with pink hints, natural lighting"
            ],

            "black_to_red": [
                "{source}dark cool complexion, add warmth → {target}warm rosy skin tone, healthy flush, maintain facial identity",
                "{source}shadowy appearance, add blood circulation → {target}naturally flushed complexion, warm undertones",
                "{source}dark dull skin, enhance vitality → {target}healthy red undertones, improve circulation appearance"
            ],

            # White → 其他 (苍白增色彩)
            "white_to_yellow": [
                "{source}pale bloodless skin, add warmth → {target}warm golden healthy complexion, hint of sunshine",
                "{source}sickly pale appearance, add vitality → {target}healthy yellow undertone, natural glow",
                "{source}white pale complexion, warm up tone → {target}balanced golden skin, add life color"
            ],

            "white_to_red": [
                "{source}pale colorless skin, add blood flow → {target}naturally rosy complexion, healthy circulation",
                "{source}ghostly white appearance, add vitality → {target}flushed healthy skin, pink undertones",
                "{source}anemic pale complexion, enhance color → {target}rosy fresh appearance, natural redness"
            ],

            "white_to_black": [
                "{source}overly pale skin, add depth → {target}darker healthy complexion, add shadow depth",
                "{source}white washed appearance, increase tone → {target}naturally dark skin, maintain health",
                "{source}pale weak complexion, add strength → {target}deeper healthy tone, enhanced vitality"
            ],

            # Yellow → 其他 (去黄增纯净)
            "yellow_to_white": [
                "{source}sallow yellow complexion, remove yellowness → {target}clean fair skin, purified appearance",
                "{source}unhealthy yellow tone, correct color → {target}bright white complexion, remove sallowness",
                "{source}muddy yellow skin, clarify tone → {target}clear healthy white skin, refined look"
            ],

            "yellow_to_red": [
                "{source}sallow yellow complexion, add life → {target}healthy rosy skin, improve circulation",
                "{source}sallow muddy appearance, add vitality → {target}naturally flushed complexion, blood flow",
                "{source}unhealthy yellow, correct to natural → {target}rosy healthy tone, balanced color"
            ],

            "yellow_to_black": [
                "{source}sallow yellow tone, deepen color → {target}healthy dark complexion, remove sallowness",
                "{source}muddy yellow skin, increase depth → {target}naturally darker skin, enhanced vitality",
                "{source}unhealthy yellowish, add richness → {target}deep healthy tone, color correction"
            ],

            # Red → 其他 (去红增平静)
            "red_to_white": [
                "{source}overly flushed red skin, calm color → {target}fair peaceful complexion, reduce redness",
                "{source}excessive red tones, normalize → {target}healthy white skin, balanced appearance",
                "{source}over-flushed complexion, soothe → {target}calm fair complexion, natural color"
            ],

            "red_to_yellow": [
                "{source}excessive redness, warm to golden → {target}healthy yellow undertone, calm flush",
                "{source}overly flushed appearance, tone down → {target}warm golden complexion, natural glow",
                "{source}too much red color, balance → {target}pleasing yellow skin, harmony restored"
            ],

            "red_to_black": [
                "{source}over-flushed redness, deepen tone → {target}healthy dark complexion, reduce redness",
                "{source}excessive red flush, add depth → {target}naturally darker skin, color balance",
                "{source}too much facial red, enhance shadow → {target}rich deeper tone, calm appearance"
            ]
        }

        # 反面prompt
        self.negative_prompts = (
            "over-saturated colors, unnatural skin transformation, artificial smoothness,plastic appearance, loss of facial features, identity change, harsh color shifts,sickly appearance, unrealistic skin tone, distorted facial structure"
        )

        # 转换强度配置
        self.conversion_strength = {
            "standard": 0.55,     # 标准转换强度
            "conservative": 0.4,  # 保守转换，更自然
            "aggressive": 0.7     # 激进转换，变化明显
        }

        self.pipeline = self._load_pipeline()
        self.stats = defaultdict(int)

    def _load_pipeline(self):
        """加载Diffusion pipeline"""
        print(f"正在加载模型: {self.model_id}")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)

        # RTX4060优化
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        return pipe

    def analyze_dataset_distribution(self, dataset_path):
        """分析数据集分布"""
        distribution = defaultdict(int)

        # 遍历所有图片
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # 从文件名中提取类别信息
                    filename = file.lower()

                    if 'black' in filename:
                        distribution['black'] += 1
                    elif 'white' in filename:
                        distribution['white'] += 1
                    elif 'yellow' in filename:
                        distribution['yellow'] += 1
                    elif 'red' in filename:
                        distribution['red'] += 1
                    else:
                        # 如果没找到明确的标签，尝试图片内容分析（简化版）
                        file_path = os.path.join(root, file)
                        predicted_type = self._predict_skin_tone(file_path)
                        distribution[predicted_type] += 1

        return dict(distribution)

    def _predict_skin_tone(self, image_path):
        """简单预测面色类型 - 基于图像分析"""
        # 这里可以实现更复杂的图像分析
        # 目前使用简单的文件名匹配
        filename = os.path.basename(image_path).lower()

        if 'black' in filename:
            return 'black'
        elif 'white' in filename:
            return 'white'
        elif 'yellow' in filename:
            return 'yellow'
        elif 'red' in filename:
            return 'red'
        else:
            return 'general'  # 默认

    def get_conversion_plan(self, current_distribution):
        """生成转换计划"""
        plan = []

        for skin_type, current_count in current_distribution.items():
            target_count = self.target_config.get(skin_type, 0)

            if current_count < target_count:
                # 需要增加此类别
                shortage = target_count - current_count
                plan.append({
                    'target_type': skin_type,
                    'needed_count': shortage,
                    'operation': 'generate'
                })
            elif current_count > target_count:
                # 此类别过多，可以作为转换源
                excess = current_count - target_count
                plan.append({
                    'source_type': skin_type,
                    'available_count': excess,
                    'operation': 'source'
                })

        return self._optimize_conversion_plan(plan)

    def _optimize_conversion_plan(self, plan):
        """优化转换计划"""
        sources = [item for item in plan if item.get('operation') == 'source']
        targets = [item for item in plan if item.get('operation') == 'generate']

        conversions = []

        for target in targets:
            remaining_needed = target['needed_count']

            for source in sources:
                if remaining_needed <= 0:
                    break

                available = source.get('available_count', 0)
                if available > 0:
                    actual_convert = min(available, remaining_needed)

                    conversions.append({
                        'from_type': source['source_type'],
                        'to_type': target['target_type'],
                        'count': actual_convert,
                        'conversion_key': f"{source['source_type']}_to_{target['target_type']}"
                    })

                    remaining_needed -= actual_convert
                    source['available_count'] -= actual_convert

        return conversions

    def convert_image(self, source_image_path, target_type, conversion_key, output_path, variation_index=0):
        """转换单个图像 - 支持多变化版本"""

        # 读取源图像
        source_image = Image.open(source_image_path).convert("RGB")

        # 获取可用的prompt列表并随机选择一个
        prompt_options = self.conversion_prompts.get(conversion_key, [])
        if not prompt_options:
            prompt_options = [f"transform {conversion_key.replace('_to_', ' to ')} skin tone, maintain facial features"]

        # 随机选择prompt并填充方向标签
        prompt_template = random.choice(prompt_options)
        prompt_filled = prompt_template.format(
            source="dark " if conversion_key.startswith("black") else ("pale " if conversion_key.startswith("white") else ("sallow " if conversion_key.startswith("yellow") else "flushed ")),
            target="warm golden " if target_type == "yellow" else ("fair " if target_type == "white" else ("rosy " if target_type == "red" else "healthy dark "))
        )

        # 添加关键短语确保身份保持
        final_prompt = f"{prompt_filled}, same facial identity, maintain recognizable features, realistic skin transformation"

        # 确定转换强度和步数
        strength = self.conversion_strength['standard']
        num_steps = 35

        # 针对特定转换优化强度
        if 'red' in conversion_key and 'to_white' in conversion_key:
            strength = 0.45  # 去红需要温和处理
        elif 'black' in conversion_key:
            strength = 0.5   # 去暗需要适度强度
        elif 'white' in conversion_key:
            strength = 0.55  # 苍白增色彩需要较强转换
        elif 'yellow' in conversion_key:
            strength = 0.4   # 去黄需要温和处理

        # 根据变化版本微调强度，增加多样性
        variation_offset = (variation_index % 3 - 1) * 0.08  # -0.08, 0, +0.08
        strength = max(0.2, min(0.8, strength + variation_offset))

        print(f"转换: {conversion_key}")
        print(f"Prompt: {final_prompt}")
        print(f"Strength: {strength:.3f}")

        # 生成转换后的图像
        with torch.no_grad():
            result = self.pipeline(
                prompt=final_prompt,
                image=source_image,
                strength=strength,
                guidance_scale=5.8,  # 稍低于之前，避免过度处理
                negative_prompt=self.negative_prompts,
                num_inference_steps=num_steps
            )

        # 保存结果
        result.images[0].save(output_path)

        self.stats['total_conversions'] += 1
        self.stats[f'{conversion_key}'] += 1

        return output_path

    def balance_dataset(self, source_dir, output_dir):
        """平衡数据集"""
        print("=== 开始中医数据集平衡 ===")

        # 1. 分析当前分布
        print(f"1. 分析数据集分布: {source_dir}")
        current_dist = self.analyze_dataset_distribution(source_dir)
        print(f"当前分布: {current_dist}")
        print(f"目标分布: {self.target_config}")

        # 2. 生成转换计划
        print("\n2. 生成转换计划...")
        conversion_plan = self.get_conversion_plan(current_dist)
        print(f"转换计划: {conversion_plan}")

        if not conversion_plan:
            print("数据集已经平衡，无需转换！")
            return

        # 3. 创建输出目录结构
        print(f"\n3. 创建输出目录: {output_dir}")
        for skin_type in self.target_config.keys():
            os.makedirs(os.path.join(output_dir, skin_type), exist_ok=True)

        # 4. 执行转换
        print(f"\n4. 开始执行转换...")

        # 获取源文件列表（按类型分组）
        source_files_by_type = defaultdict(list)

        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    skin_type = self._predict_skin_tone(file_path)
                    source_files_by_type[skin_type].append(file_path)

        # 5. 复制原文件到输出目录
        print(f"\n5. 复制源文件...")
        files_copied = 0
        for skin_type, file_list in source_files_by_type.items():
            for file_path in file_list:
                base_name = os.path.basename(file_path)
                shutil.copy2(file_path, os.path.join(output_dir, skin_type, base_name))
                files_copied += 1

        print(f"已复制 {files_copied} 个源文件")

        # 6. 执行转换
        print(f"\n6. 执行Diffusion转换...")
        conversions_done = 0

        for conversion in conversion_plan:
            from_type = conversion['from_type']
            to_type = conversion['to_type']
            count = conversion['count']
            conversion_key = conversion['conversion_key']

            print(f"\n执行转换: {from_type} → {to_type} (目标数量: {count})")

            # 获取可用的源文件
            available_files = [f for f in source_files_by_type[from_type]
                            if not any(f.endswith(x) for x in ['_converted.png', '_balancing.png'])]

            if len(available_files) < count:
                print(f"  警告: 可用文件不足 ({len(available_files)} < {count})")
                count = len(available_files)

            # 随机选择文件进行转换
            selected_files = random.sample(available_files, count)

            for i, source_file in enumerate(selected_files):
                # 构建输出文件名
                base_name = os.path.splitext(os.path.basename(source_file))[0]
                output_filename = f"{base_name}_balancing_{to_type}.png"
                output_path = os.path.join(output_dir, to_type, output_filename)

                print(f"  [{i+1}/{count}] 转换 {os.path.basename(source_file)}...")

                try:
                    self.convert_image(source_file, to_type, conversion_key, output_path)
                    conversions_done += 1
                except Exception as e:
                    print(f"    转换失败: {e}")

        # 7. 最终统计
        print(f"\n=== 转换完成 ===")
        print(f"总转换数量: {conversions_done}")
        print(f"详细统计: {dict(self.stats)}")

        # 8. 验证最终分布
        print(f"\n验证最终分布...")
        final_dist = self.analyze_dataset_distribution(output_dir)
        print(f"最终分布: {final_dist}")

        return {
            'original_distribution': current_dist,
            'final_distribution': final_dist,
            'conversions_performed': conversions_done,
            'conversion_plan': conversion_plan,
            'statistics': dict(self.stats)
        }

# 使用示例
if __name__ == "__main__":
    # 创建平衡器
    balancer = TCMDatasetBalancer(
        target_config={
            "black": 100,
            "white": 100,
            "yellow": 100,
            "red": 100
        }
    )

    # 执行数据集平衡
    source_directory = "D:/TCMinspection/imagesource/faces_original"
    output_directory = "D:/TCMinspection/imagesource/faces_balanced"

    try:
        result = balancer.balance_dataset(source_directory, output_directory)

        # 保存转换记录
        with open(os.path.join(output_directory, "conversion_log.json"), 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'source_dir': source_directory,
                'output_dir': output_directory,
                'results': result
            }, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()