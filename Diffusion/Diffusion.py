import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# -------------------------------
# 1. 加载模型
# -------------------------------
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")  # 如果没有GPU，可以改成 "cpu"

# -------------------------------
# 2. 读取输入图像
# -------------------------------
input_path = "D:/TCMinspection/imagesource/faces_original/172_black.png"
image = Image.open(input_path).convert("RGB")

# -------------------------------
# 3. 设置prompt
# -------------------------------
# 原图是标签 "black"，我们想生成一个变体，比如肤色更健康/偏黄/偏红
prompt = "a portrait of a healthy person with warm skin tone, high quality, detailed"
negative_prompt = "blurry, deformed, distorted, ugly, unrealistic, bad anatomy, watermark"

# -------------------------------
# 4. 生成新图像
# -------------------------------
images = pipe(
    prompt=prompt,
    image=image,
    strength=0.6,   # 控制改动幅度：0.3=微调，0.8=大改
    guidance_scale=7.5,  # 越大越贴近 prompt
    negative_prompt=negative_prompt,
    num_inference_steps=50
).images

# -------------------------------
# 5. 保存结果
# -------------------------------
output_path = "output/face_black_aug.png"
images[0].save(output_path)

print(f"新图像已保存到 {output_path}")
