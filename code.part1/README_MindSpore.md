# MindSpore 重构版本说明

本目录包含使用华为MindSpore框架重构的原有PyTorch代码，保持了原有功能不变。

## 文件说明

### 🧠 监督自编码器 (Supervised Autoencoder)
- **原文件**: `Autoencoder.py`
- **重构文件**: `Autoencoder_mindspore.py`

**功能保持不变**:
- 维度缩减：4096 → 128
- 监督学习：同时优化重构损失和分类损失
- 网络结构：编码器4096→1024→256→128，解码器128→256→1024→4096
- 分类器：128→5类别

**主要改动**:
- PyTorch → MindSpore API替换
- `torch.nn.Module` → `mindspore.nn.Cell`
- `torch.optim.Adam` → `mindspore.nn.Adam`
- `torch.save/load` → `mindspore.save/load_checkpoint`

### 📸 面部区域分割与特征提取
- **原文件**: `分割面部区域未修改CNN版.py`
- **重构文件**: `分割面部区域_mindspore.py`

**功能保持不变**:
- 基于dlib的68点面部关键点检测
- 提取9个TCM诊断关键面部区域
- CNN特征提取（VGG16）
- 手工特征提取（RGB、HSV、Lab、YCbCr、GLCM）

**主要改动**:
- PyTorch VGG16 → MindSpore VGG16
- `torchvision.transforms` → `mindspore.dataset.vision`
- `torch.Tensor` → `mindspore.Tensor`
- 保持相同的特征维度和处理流程

## 运行要求

```bash
# 安装MindSpore（GPU版本）
pip install mindspore-gpu

# 依赖库
pip install opencv-python dlib pillow scikit-learn scikit-image
```

## 使用示例

### 训练自编码器
```bash
cd code.part1
python Autoencoder_mindspore.py
```

### 面部特征提取
```bash
cd code.part1
python 分割面部区域_mindspore.py
```

### 运行测试
```bash
cd code.part1
python test_mindspore_implementations.py
```

## 输出文件

- `autoencoder_model_mindspore.ckpt`: MindSpore格式的自编码器模型
- `newencoder_mindspore.npy`: 提取的CNN特征
- `label_mindspore.npy`: 对应的标签

## 性能对比

重构后的MindSpore版本保持与原始PyTorch版本相同的：
- ✅ 网络结构和参数量
- ✅ 特征提取精度
- ✅ 输出维度和格式
- ✅ TCM面部诊断区域定义
- ✅ 分类性能（理论上）

## 注意事项

1. **模型加载**: MindSpore使用`.ckpt`格式保存模型，与PyTorch的`.pth`格式不兼容
2. **数据处理**: 保持归一化参数不变(ImageNet均值标准差)
3. **面部检测**: 仍需使用dlib Shape Predictor进行68点关键点检测
4. **GPU支持**: 需要配置正确的CUDA环境支持MindSpore GPU版本

## 测试验证

运行 `test_mindspore_implementations.py` 可以验证：
- 模型结构一致性
- 特征提取兼容性
- 数据处理正确性
- 前向传播功能正常性

重构完成！🎉 项目现已支持华为MindSpore框架。