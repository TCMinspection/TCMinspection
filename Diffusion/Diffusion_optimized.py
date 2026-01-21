import torch
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
import time

# 中医面色类型与prompt映射 - 基于中医五色理论
tcm_skin_tone_prompts = {
    "black": "healthy natural skin tone removing darkish appearance, balanced kidney-energy complexion, warm undertones, traditional Chinese medicine healthy kidney appearance, detailed realistic skin texture",

    "white": "healthy rosy complexion replacing pale appearance, nourished blood-energy skin tone, subtle pink undertones, traditional Chinese medicine qi-blood balance, natural skin vitality",

    "yellow": "warm healthy complexion with balanced spleen-energy, natural golden skin tone removing sallowness, traditional Chinese medicine spleen-harmony appearance, healthy glow",

    "red": "naturally balanced complexion reducing excessive redness, harmonized heart-energy skin tone, traditional Chinese medicine heart-balance appearance, calm healthy facial color",

    "general_health": "naturally balanced skin tone with subtle healthy variations, optimal facial complexion according to traditional Chinese medicine, harmonized skin color with life vitality"
}

# 反面prompt - 避免不自然的中医面色
negative_prompt_base = "over-saturated colors, unnatural skin tone, excessive artificial smoothness, plastic doll appearance, loss of skin detail, harsh color changes, artificial filter effect, sickly appearance"

# 中医面色类型参数配置 - 根据中医理论调整不同面色的处理强度
tcm_generation_configs = {
    "black": {
        "strength": 0.4,      # 温和调整，避免过度改变
        "guidance_scale": 6.0,  # 中等引导，保持自然
        "prompt_focus": "warmth and vitality"
    },
    "white": {
        "strength": 0.5,      # 需要更多调整
        "guidance_scale": 6.5,  # 稍强引导，增加血色
        "prompt_focus": "nourished and vital"
    },
    "yellow": {
        "strength": 0.35,     # 轻微调整，保持自然
        "guidance_scale": 5.5,  # 较低引导，避免过度
        "prompt_focus": "balanced and harmonious"
    },
    "red": {
        "strength": 0.3,      # 最小调整，降红
        "guidance_scale": 5.0,  # 最低引导，自然降红
        "prompt_focus": "calm and balanced"
    },
    "general_health": {
        "strength": 0.4,
        "guidance_scale": 6.0,
        "prompt_focus": "naturally healthy"
    }
}

def load_facechain_pipeline():
    """加载FaceChain模型 - RTX4060优化"""
    print("正在加载FaceChain模型...")
    start_time = time.time()

    # 使用BAAI/FaceChain-Face-Portrait - 最适合中医面部生成
    model_id = "BAAI/FaceChain-Face-Portrait"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to("cuda")

    # RTX4060优化设置
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    # 使用更快的scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    load_time = time.time() - start_time
    print(f"模型加载完成，用时: {load_time:.2f}秒")
    print(f"显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

    return pipe

def determine_skin_tone_type(original_filename):
    """根据文件名判断中医面色类型"""
    filename_lower = original_filename.lower()

    if "black" in filename_lower:
        return "black"
    elif "white" in filename_lower:
        return "white"
    elif "yellow" in filename_lower:
        return "yellow"
    elif "red" in filename_lower:
        return "red"
    else:
        return "general_health"  # 默认

def generate_tcm_face_variation(pipe, image_path, output_dir="output", skin_tone_type=None):
    """生成中医面色变体"""

    # 读取输入图像
    image = Image.open(image_path).convert("RGB")

    # 确定面色类型
    if skin_tone_type is None:
        skin_tone_type = determine_skin_tone_type(os.path.basename(image_path))

    print(f"识别的面色类型: {skin_tone_type}")

    # 获取配置
    config = tcm_generation_configs.get(skin_tone_type, tcm_generation_configs["general_health"])
    base_prompt = tcm_skin_tone_prompts.get(skin_tone_type, tcm_skin_tone_prompts["general_health"])

    # 构建完整prompt
    full_prompt = f"{base_prompt}, {config['prompt_focus']}, high quality, detailed facial features, natural lighting"

    print(f"使用的prompt: {full_prompt}")
    print(f"生成参数 - strength: {config['strength']}, guidance_scale: {config['guidance_scale']}")

    # 生成图像
    print("开始生成...")
    start_time = time.time()

    with torch.no_grad():
        result = pipe(
            prompt=full_prompt,
            image=image,
            strength=config['strength'],
            guidance_scale=config['guidance_scale'],
            negative_prompt=negative_prompt_base,
            num_inference_steps=30,
            eta=0.0
        )

    generation_time = time.time() - start_time
    print(f"生成完成，用时: {generation_time:.2f}秒")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_tcm_{skin_tone_type}_enhanced.png")

    result.images[0].save(output_path)
    print(f"结果已保存到: {output_path}")

    return output_path

def compare_with_original(original_path, generated_path):
    """简单对比生成质量"""
    original = Image.open(original_path)
    generated = Image.open(generated_path)

    print(f"原图尺寸: {original.size}")
    print(f"生成图尺寸: {generated.size}")
    print(f"生成模式: FaceChain + TCM优化")

def main():
    """主函数"""
    # 加载模型
    pipe = load_facechain_pipeline()

    # 处理多个图像示例
    test_images = [
        "D:/TCMinspection/imagesource/faces_original/172_black.png",
        "D:/TCMinspection/imagesource/faces_original/173_white.png",
        "D:/TCMinspection/imagesource/faces_original/174_yellow.png"
    ]

    print("\n=== 开始中医面色增强生成 ===")

    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n处理图像: {image_path}")
            try:
                generated_path = generate_tcm_face_variation(pipe, image_path)
                compare_with_original(image_path, generated_path)
            except Exception as e:
                print(f"处理失败: {e}")
        else:
            print(f"文件不存在: {image_path}")

    print("\n=== 所有处理完成 ===")
    print(f"总显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

if __name__ == "__main__":
    main()