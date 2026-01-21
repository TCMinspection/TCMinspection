
import os
# HF_TOKEN should be set as environment variable for security
# os.environ["HF_TOKEN"] = "your-huggingface-token-here"
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from PIL import Image
import shutil
import json
from collections import defaultdict
import glob
import time
import random
import cv2
import numpy as np



class IntelligentTCMDatasetBalancer:
    """智能中医面色数据集平衡器 - 基于颜色相似度的最佳转换策略"""

    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", target_config=None, device="cuda"):
        """
        初始化智能平衡器

        Args:
            model_id: 使用的diffusion模型
            target_config: 目标分布配置，例如：{"black": 150, "white": 100, "yellow": 100, "red": 100, "cyan": 80}
            device: 运行设备
        """
        self.model_id = model_id
        self.device = device

        # 支持6种面色类型，包括新增的青色(cyan)
        self.target_config = target_config or {
            "black": 150,   # 目标：150张 - 可以作为源
            "white": 100,   # 目标：100张 - 可以作为源
            "yellow": 100,  # 目标：100张
            "red": 100,     # 目标：100张 - 可以作为源
            "cyan": 80      # 目标：80张 - 新增青色
        }

        # 颜色相似度矩阵 - 基于中医五色理论的转换优先级
        # 高数值表示颜色相近，转换更容易成功
        self.color_similarity_matrix = {
            #        black  white  yellow  red   cyan
            "black":  [1.0,   0.3,   0.8,   0.2,  0.9],  # black最适合转yellow/cyan
            "white":  [0.3,   1.0,   0.6,   0.4,  0.5],  # white适合转yellow
            "yellow": [0.8,   0.6,   1.0,   0.5,  0.7],  # yellow转black/white/cyan
            "red":    [0.2,   0.4,   0.5,   1.0,  0.3],  # red适合转white/yellow
            "cyan":   [0.9,   0.5,   0.7,   0.3,  1.0]   # cyan适合转black/yellow
        }

        # 面色色调的RGB近似值（用于颜色距离计算）
        self.color_rgb_approx = {
            "black":  [45, 35, 30],   # 暗黑色
            "white":  [240, 230, 220], # 苍白色
            "yellow": [200, 180, 120], # 萎黄色
            "red":    [180, 120, 110], # 红色
            "cyan":   [120, 150, 160]  # 青色（介于蓝绿之间）
        }

        # 用户要求的优选转换路径
        self.preferred_conversions = {
            "black_to_yellow": {"priority": 0.95, "match_5_elements": True},
            "black_to_cyan": {"priority": 0.90, "match_5_elements": True},
            "red_to_white": {"priority": 0.85, "match_5_elements": True},
            "white_to_yellow": {"priority": 0.80, "match_5_elements": True},
            "yellow_to_black": {"priority": 0.75, "match_5_elements": True}
        }

        # 智能转换prompt映射 - 基于中医五行理论和颜色相似度
        self.smart_conversion_prompts = {
            # Black → 相近色系 (高优先级)
            "black_to_yellow": [
                "{source}dark dull complexion, gently transition to {target}warm golden healthy skin tone, maintain natural facial structure",
                "{source}grayish dark skin, lift shadow undertones to {target}sunny warm complexion, preserve facial identity",
                "{source}overly dark skin tone, transform to {target}healthy yellow undertone with balanced appearance"
            ],

            "black_to_cyan": [
                "{source}dark shadowy skin, transition to {target}natural bluish-green healthy complexion, maintain facial features",
                "{source}deep dull complexion, convert to {target}cool cyan skin tone with balanced energy",
                "{source}dark cool appearance, transform to {target}fresh cyan complexion with vitality"
            ],

            # Red → 相近色系
            "red_to_white": [
                "{source}overly flushed complexion, calm to {target}fair peaceful skin tone, reduce redness naturally",
                "{source}excessive red tones, transition to {target}healthy white complexion with balanced appearance",
                "{source}flushed facial color, transform to {target}pure white skin, maintain natural glow"
            ],

            "red_to_yellow": [
                "{source}overly red skin, warm to {target}golden yellow healthy complexion, natural transition",
                "{source}excessive facial redness, convert to {target}pleasing yellow tone with harmony",
                "{source}intense red complexion, transform to {target}warm yellow with vitality balance"
            ],

            # White → 相近色系
            "white_to_yellow": [
                "{source}pale fair skin, add warm undertones to {target}golden healthy complexion",
                "{source}light white complexion, transition to {target}warm sunny skin tone naturally",
                "{source}pale dull appearance, enhance to {target}healthy yellow with life energy"
            ],

            # Yellow → 相近色系
            "yellow_to_black": [
                "{source}sallow yellow complexion, deepen to {target}rich dark healthy skin tone",
                "{source}muddy yellow appearance, transition to {target}deep balanced dark complexion",
                "{source}unhealthy yellowish tone, transform to {target}natural dark skin with vitality"
            ],

            "yellow_to_cyan": [
                "{source}warm yellow skin, cool to {target}bluish-green healthy complexion naturally",
                "{source}golden yellow appearance, convert to {target}cool cyan tone with balance",
                "{source}yellowish complexion, transform to {target}refreshing cyan with harmony"
            ]
        }

        # 转换强度配置（基于颜色相似度优化）
        self.conversion_strength_by_similarity = {
            "high": 0.45,   # 颜色非常相近 - 温和转换
            "medium": 0.55, # 颜色中等相近 - 标准转换
            "low": 0.65     # 颜色差异较大 - 略强转换
        }

        # 距离到相似度的映射
        self.distance_to_strength = {
            (0.0, 0.3): "high",
            (0.3, 0.6): "medium",
            (0.6, 1.0): "low"
        }

        # 反面提示词 - 避免不自然的转换效果
        self.negative_prompts = (
            "over-saturated colors, unnatural skin transformation, artificial smoothness, "
            "plastic appearance, loss of facial features, identity change, harsh color shifts, "
            "unrealistic skin tone, distorted facial structure, "
            "exaggerated skin texture, artificial lighting, over-processed appearance"
        )

        self.pipeline = self._load_pipeline()
        self.stats = defaultdict(int)
        self.color_cache = {}  # 缓存计算过的颜色特征

    def _load_pipeline(self):
        """加载Diffusion pipeline"""
        print(f"正在加载智能模型: {self.model_id}")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir="D:/.cache/huggingface/hub"
        ).to(self.device)

        # RTX4060优化
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        return pipe

    def analyze_dataset_distribution(self, dataset_path):
        """分析数据集分布"""
        distribution = defaultdict(int)

        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filename = file.lower()

                    skin_types = ['black', 'white', 'yellow', 'red', 'cyan']
                    for skin_type in skin_types:
                        if f'_{skin_type}.' in filename:  # 优先匹配 _color. 模式
                            distribution[skin_type] += 1
                            break
                    else:
                        # 如果没找到明确标签，使用图像分析
                        file_path = os.path.join(root, file)
                        predicted_type = self._intelligent_predict_skin_tone(file_path)
                        distribution[predicted_type] += 1

        return dict(distribution)

    def _intelligent_predict_skin_tone(self, image_path):
        """智能预测面色类型 - 先检查文件名，再使用图像分析"""
        if image_path in self.color_cache:
            return self.color_cache[image_path]

        # 优先从文件名中提取label
        filename = os.path.basename(image_path).lower()
        skin_types = ['black', 'white', 'yellow', 'red', 'cyan']
        for skin_type in skin_types:
            if f'_{skin_type}.' in filename:
                self.color_cache[image_path] = skin_type
                return skin_type

        try:
            image = cv2.imread(image_path)
            if image is None:
                return 'general'

            # 转换到LAB色彩空间进行更准确的皮肤分析
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

            # 提取皮肤区域（简化版的皮肤检测）
            l_channel = lab_image[:,:,0]
            a_channel = lab_image[:,:,1]
            b_channel = lab_image[:,:,2]

            # 基本的皮肤检测阈值
            skin_mask = (l_channel > 60) & (l_channel < 200) & (a_channel > 130) & (a_channel < 180) & (b_channel > 130) & (b_channel < 180)

            if np.sum(skin_mask) < 100:  # 皮肤区域太小
                result = 'general'
            else:
                # 计算平均LAB值
                avg_l = np.mean(l_channel[skin_mask])
                avg_a = np.mean(a_channel[skin_mask])
                avg_b = np.mean(b_channel[skin_mask])

                # 基于LAB值进行智能分类
                result = self._classify_by_lab_values(avg_l, avg_a, avg_b)

            self.color_cache[image_path] = result
            return result

        except Exception:
            return 'general'

    def _classify_by_lab_values(self, l, a, b):
        """基于LAB颜色值智能分类面色"""
        # 定义各类面色的LAB参考值
        reference_values = {
            'black': [60, 140, 140],    # 暗黑色 - 低L值
            'white': [180, 140, 140],   # 苍白色 - 高L值
            'yellow': [140, 140, 160],  # 萎黄色 - 高B值
            'red': [140, 160, 140],     # 红色 - 高A值
            'cyan': [120, 130, 150]     # 青色 - 适中A值，高B值
        }

        # 计算到各类参考值的欧氏距离
        distances = {}
        for skin_type, (ref_l, ref_a, ref_b) in reference_values.items():
            distance = np.sqrt((l - ref_l)**2 + (a - ref_a)**2 + (b - ref_b)**2)
            distances[skin_type] = distance

        # 选择距离最小的类别
        return min(distances, key=distances.get)

    def calculate_color_similarity(self, source_type, target_type):
        """计算两种面色类型之间的相似度"""
        # 使用预定义的相似度矩阵
        if source_type in self.color_similarity_matrix and target_type in self.color_similarity_matrix[source_type]:
            return self.color_similarity_matrix[source_type][list(self.color_similarity_matrix.keys()).index(target_type)]

        # 备选：使用RGB距离计算
        if source_type in self.color_rgb_approx and target_type in self.color_rgb_approx:
            rgb1 = np.array(self.color_rgb_approx[source_type])
            rgb2 = np.array(self.color_rgb_approx[target_type])

            # 标准化欧氏距离（转换为0-1范围）
            distance = np.linalg.norm(rgb1 - rgb2) / 441.67  # sqrt(255^2 * 3)
            similarity = 1.0 - min(distance, 1.0)
            return similarity

        return 0.5  # 默认相似度

    def get_conversion_priority(self, source_type, target_type):
        """获取转换优先级（综合考虑用户偏好和颜色相似度）"""
        conversion_key = f"{source_type}_to_{target_type}"

        # 首先检查用户定义的优选转换
        if conversion_key in self.preferred_conversions:
            return self.preferred_conversions[conversion_key]["priority"]

        # 然后计算基于颜色相似度的优先级
        similarity = self.calculate_color_similarity(source_type, target_type)

        # 将相似度映射到0.6-0.8的范围（避免优选转换被普通转换覆盖）
        return 0.6 + 0.2 * similarity

    def generate_smart_conversion_plan(self, current_distribution):
        """基于颜色相似度生成最优转换计划"""
        print("🎨 生成智能颜色相似度转换计划...")

        # 分析当前状态
        shortages = {}  # 需要增加的类别
        excesses = {}   # 有过多的类别
        all_skin_types = list(self.target_config.keys())

        for skin_type in all_skin_types:
            current = current_distribution.get(skin_type, 0)
            target = self.target_config[skin_type]

            if current < target:
                shortages[skin_type] = target - current
            elif current > target:
                excesses[skin_type] = current - target

        print(f"📊 当前分布: {dict(current_distribution)}")
        print(f"🎯 目标分布: {self.target_config}")
        print(f"📈 短缺类别: {shortages}")
        print(f"📉 过剩类别: {excesses}")

        # 生成所有可能的转换路径及其优先级
        conversion_candidates = []

        for source_type, excess_count in excesses.items():
            for target_type, shortage_count in shortages.items():
                if source_type != target_type:  # 不能自己转自己
                    priority = self.get_conversion_priority(source_type, target_type)

                    conversion_candidates.append({
                        'from_type': source_type,
                        'to_type': target_type,
                        'priority': priority,
                        'similarity': self.calculate_color_similarity(source_type, target_type),
                        'max_possible': min(excess_count, shortage_count)
                    })

        # 按优先级排序（高优先级优先）
        conversion_candidates.sort(key=lambda x: x['priority'], reverse=True)

        print(f"🔍 找到 {len(conversion_candidates)} 种可能的转换路径")
        for candidate in conversion_candidates[:10]:  # 只显示前10个
            print(f"  {candidate['from_type']}→{candidate['to_type']}: 优先级={candidate['priority']:.2f}, 相似度={candidate['similarity']:.2f}")

        # 生成最终转换计划
        final_plan = []
        remaining_shortages = shortages.copy()
        remaining_excesses = excesses.copy()

        for candidate in conversion_candidates:
            if remaining_shortages.get(candidate['to_type'], 0) > 0 and remaining_excesses.get(candidate['from_type'], 0) > 0:
                # 计算实际需要转换的数量
                convert_count = min(
                    candidate['max_possible'],
                    remaining_shortages[candidate['to_type']],
                    remaining_excesses[candidate['from_type']]
                )

                if convert_count > 0:
                    final_plan.append({
                        'from_type': candidate['from_type'],
                        'to_type': candidate['to_type'],
                        'count': convert_count,
                        'priority': candidate['priority'],
                        'similarity': candidate['similarity'],
                        'conversion_key': f"{candidate['from_type']}_to_{candidate['to_type']}"
                    })

                    # 更新剩余数量
                    remaining_shortages[candidate['to_type']] -= convert_count
                    remaining_excesses[candidate['from_type']] -= convert_count

        # 🔧 扩增逻辑：当所有类别都不足时，使用数据扩增策略
        if remaining_shortages and sum(remaining_excesses.values()) == 0:
            print("🔧 所有类别都不足，采用智能数据扩增策略...")

            # 过滤掉完全没有数据的类别
            valid_current = {k: v for k, v in current_distribution.items() if v > 0}
            if not valid_current:
                print("⚠️  警告：没有可用的源数据进行扩增")
                return final_plan, shortages, excesses

            # 找出当前数量最多的类别作为最佳扩增源
            max_count = max(valid_current.values())
            potential_sources = [stype for stype, count in valid_current.items() if count == max_count]

            if potential_sources:
                # 选择转换优先级最高的源类别进行扩增
                best_source = max(potential_sources,
                                key=lambda x: max(self.get_conversion_priority(x, target)
                                                for target in remaining_shortages.keys()))

                for target_type, remaining_shortage in remaining_shortages.items():
                    if best_source != target_type:
                        # 计算扩增倍数，限制在给定范围内（避免过拟合）
                        multiplier = min(3, max(1, int(remaining_shortage / valid_current[best_source] + 0.8)))
                        expansion_count = min(remaining_shortage, multiplier * valid_current[best_source])

                        if expansion_count > 0:
                            final_plan.append({
                                'from_type': best_source,
                                'to_type': target_type,
                                'count': expansion_count,
                                'priority': self.get_conversion_priority(best_source, target_type),
                                'similarity': self.calculate_color_similarity(best_source, target_type),
                                'conversion_key': f"{best_source}_to_{target_type}",
                                'strategy': 'augmentation',
                                'multiplier': multiplier
                            })

                print(f"🎯 扩增策略选择：从 '{best_source}' 扩增到多个目标类别")
                for plan in final_plan[-min(5, len(remaining_shortages)):]:  # 显示最后几个扩增计划
                    if plan.get('strategy') == 'augmentation':
                        print(f"   {plan['from_type']}→{plan['to_type']}: {plan['count']}张 x{plan['multiplier']}倍 (扩增模式)")

        return final_plan, shortages, excesses

    def convert_image(self, source_image_path, target_type, conversion_key, output_path, variation_index=0):
        """智能转换单个图像 - 基于颜色相似度优化"""
        # 读取源图像
        source_image = Image.open(source_image_path).convert("RGB")

        # 获取智能转换prompt
        prompt_options = self.smart_conversion_prompts.get(conversion_key, [])
        if not prompt_options:
            prompt_options = [f"transform {conversion_key.replace('_to_', ' to ')} skin tone, maintain facial features"]

        # 随机选择prompt
        prompt_template = random.choice(prompt_options)
        similarity = self.calculate_color_similarity(conversion_key.split('_to_')[0], target_type)

        # 根据相似度调整prompt描述
        if similarity > 0.8:
            difficulty_desc = "smoothly transition"
        elif similarity > 0.5:
            difficulty_desc = "naturally convert"
        else:
            difficulty_desc = "transform"

        final_prompt = f"{prompt_template}, {difficulty_desc} while maintaining facial identity, realistic skin tone change"

        # 根据颜色相似度确定转换强度
        if similarity > 0.8:
            strength = self.conversion_strength_by_similarity["high"]
        elif similarity > 0.5:
            strength = self.conversion_strength_by_similarity["medium"]
        else:
            strength = self.conversion_strength_by_similarity["low"]

        # 添加变化偏移
        variation_offset = (variation_index % 3 - 1) * 0.05
        strength = max(0.3, min(0.7, strength + variation_offset))

        print(f"🎨 智能转换: {conversion_key}")
        print(f"   颜色相似度: {similarity:.2f}")
        print(f"   转换强度: {strength:.3f}")
        print(f"   提示词: {final_prompt[:80]}...")

        # 生成转换后的图像
        with torch.no_grad():
            result = self.pipeline(
                prompt=final_prompt,
                image=source_image,
                strength=strength,
                guidance_scale=6.0,  # 适中的guidance
                negative_prompt=self.negative_prompts,
                num_inference_steps=50
            )

        # 保存结果
        result.images[0].save(output_path)

        self.stats['total_conversions'] += 1
        self.stats[f'{conversion_key}'] += 1

        return output_path

    def intelligent_balance_dataset(self, source_dir, output_dir):
        """基于智能颜色相似度的数据集平衡"""
        print("=== 🧠 启动智能中医数据集平衡 ===")

        # 1. 分析当前分布
        print(f"📊 分析数据集分布: {source_dir}")
        current_dist = self.analyze_dataset_distribution(source_dir)
        print(f"📝 当前分布: {current_dist}")
        print(f"🎯 目标分布: {self.target_config}")

        # 2. 生成智能转换计划
        conversion_plan, shortages, excesses = self.generate_smart_conversion_plan(current_dist)

        if not conversion_plan:
            print("⚠️  无法生成转换计划：数据集过小，可能需要收集更多数据！")
            print("📊 当前总数据量不足，所有类别都短缺")
            return

        print(f"🚀 生成转换计划: {len(conversion_plan)} 个转换任务")
        for plan in conversion_plan:
            print(f"   {plan['from_type']} → {plan['to_type']}: {plan['count']} 张 (相似度: {plan['similarity']:.2f}, 优先级: {plan['priority']:.2f})")

        # 3. 创建输出目录结构
        print(f"📁 创建输出目录: {output_dir}")
        for skin_type in self.target_config.keys():
            os.makedirs(os.path.join(output_dir, skin_type), exist_ok=True)

        # 4. 复制原始文件
        print(f"📋 复制原始文件...")
        originals_copied = 0
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src_path = os.path.join(root, file)

                    # 确定目标类别
                    skin_type = self._intelligent_predict_skin_tone(src_path)
                    if skin_type not in self.target_config:
                        skin_type = 'general'

                    dst_path = os.path.join(output_dir, skin_type, file)
                    shutil.copy2(src_path, dst_path)
                    originals_copied += 1

        print(f"✅ 已复制 {originals_copied} 个原始文件")

        # 5. 执行智能转换
        print(f"\n🔄 开始执行智能转换...")
        conversions_done = 0

        # 为每个转换计划准备源文件列表
        available_files_by_type = defaultdict(list)
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    skin_type = self._intelligent_predict_skin_tone(file_path)
                    if skin_type in self.target_config:
                        available_files_by_type[skin_type].append(file_path)

        # 执行转换任务
        for conversion_task in conversion_plan:
            from_type = conversion_task['from_type']
            to_type = conversion_task['to_type']
            count = conversion_task['count']
            conversion_key = conversion_task['conversion_key']

            print(f"\n🎯 执行转换: {from_type} → {to_type} (目标: {count} 张)")

            # 获取可用的源文件
            available_files = available_files_by_type.get(from_type, [])
            if len(available_files) < count:
                print(f"   ⚠️  可用文件不足 ({len(available_files)} < {count})")
                count = len(available_files)

            if count == 0:
                continue

            # 随机选择文件进行转换
            selected_files = random.sample(available_files, count)

            # 从可用列表中移除已选择的文件
            for file_path in selected_files:
                if file_path in available_files_by_type[from_type]:
                    available_files_by_type[from_type].remove(file_path)

            # 执行转换
            for i, source_file in enumerate(selected_files):
                base_name = os.path.splitext(os.path.basename(source_file))[0]
                output_filename = f"{base_name}_intelligent_{to_type}.png"
                output_path = os.path.join(output_dir, to_type, output_filename)

                print(f"   [{i+1}/{count}] 转换: {os.path.basename(source_file)}...")

                try:
                    self.convert_image(source_file, to_type, conversion_key, output_path, variation_index=i)
                    conversions_done += 1
                except Exception as e:
                    print(f"   ❌ 转换失败: {e}")

        # 6. 最终结果汇总
        print(f"\n🎉 === 智能转换完成 ===")
        print(f"📈 总转换数量: {conversions_done}")
        print(f"📊 详细统计:")
        for conversion_key, count in self.stats.items():
            if conversion_key != 'total_conversions':
                print(f"   {conversion_key}: {count} 张")

        # 7. 验证最终分布
        final_dist = self.analyze_dataset_distribution(output_dir)
        print(f"\n📊 最终分布: {final_dist}")

        # 生成转换报告
        report = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'source_directory': source_dir,
            'output_directory': output_dir,
            'original_distribution': current_dist,
            'final_distribution': final_dist,
            'conversion_plan': conversion_plan,
            'total_conversions': conversions_done,
            'statistics': dict(self.stats),
            'shortages_addressed': dict(shortages),
            'excesses_utilized': dict(excesses)
        }

        # 保存转换记录
        report_path = os.path.join(output_dir, "intelligent_conversion_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n📝 详细报告已保存到: {report_path}")

        return report

# 使用示例和测试
if __name__ == "__main__":
    # 创建智能平衡器
    balancer = IntelligentTCMDatasetBalancer(
        target_config={
            "black": 150,
            "white": 120,
            "yellow": 100,
            "red": 100,
            "cyan": 80
        }
    )

    # 执行智能数据集平衡
    source_directory = "D:/TCMinspection/imagesource/faces_original"
    output_directory = "D:/TCMinspection/imagesource/faces_intelligent_balanced"

    try:
        result = balancer.intelligent_balance_dataset(source_directory, output_directory)
        print("\n✨ 智能转换任务全部完成！")

    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()