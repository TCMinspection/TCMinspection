# 本文档用于对该文件夹中的部分训练数据以及提取到的特征文件进行说明

## Autoencoder训练以及权重文件：
1. **autoencoder_model_mindspore.ckpt** : mindspore架构下的autoencoder权重文件
2. **autoencoder_model.pth**: pytorch架构下的autoencoder权重文件
3. **feature_mindspore.npy**: mindspore架构下提取用于训练autoencoder的特征文件
4. **encoder.npy**: pytorch架构下的autoencoder训练的特征文件

## CNN提取后的特征和标签文件：
1. **feature.npy**: pytorch架构下提取的特征文件
2. **feature_mindspore_class.npy**:mindspore架构下用于svm分类的特征
3. **label_mindspore_class.npy**:mindspore用于svm分类的标签
4. **label_mindspore.npy**:mindspore架构用于autoencoder训练的标签

## SVM


## 部分图片以及训练数据集

1. **imagesourece1.zip**: 所有训练数据集
2. **屏幕截图 2025-04-26 235408.png**: 测试集上效果图
3. **IMG_1786.PNG**：网页展示效果图
4. **imagesource1**: 扩增后训练数据集（）
5. **imagesoure/faces_original**: 原始数据集（训练autoencoder用）

