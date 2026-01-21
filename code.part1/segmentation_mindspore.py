import cv2
import dlib
import numpy as np
from PIL import Image
import os
from skimage.feature import graycomatrix, graycoprops
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.common import dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore import load_checkpoint, load_param_into_net

# 导入Autoencoder相关功能
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Autoencoder_mindspore import SupervisedAutoencoder, encode_features

# 设置运行环境
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 数据预处理函数
def create_data_transforms1():
    """创建第一个数据预处理管道"""
    transforms = [
        vision.Resize((224, 224)),
        vision.RandomColorAdjust(brightness=(0.5, 1),contrast=(0.4, 1)),
        vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        vision.HWC2CHW()
    ]
    return transforms


import mindspore as ms
import mindspore.nn as nn
from mindcv.models import vgg16,mobilenet_v3_large_075

# class MultiModelFeatures(nn.Cell):
#     def __init__(self, model_type='vgg16'):
#         super(MultiModelFeatures, self).__init__()
#         self.model_type = model_type

#         if model_type == 'vgg16':
#             # 使用 mindvision 加载 VGG16
#             vgg = vgg16(pretrained=True)
#             self.features = vgg.features
#             # 使用 VGG16 的分类器部分，但去掉最后一层
#             self.classifier = nn.SequentialCell([
#                 nn.Dense(25088, 4096),
#                 nn.ReLU(),
#                 nn.Dropout(0.5),
#                 nn.Dense(4096, 4096),
#                 nn.ReLU(),
#                 nn.Dropout(0.5)
#             ])
#         elif model_type == 'mobilenet':
#             mobilenet = mobilenet_v3_large_075(pretrained=True)
#             self.features = mobilenet.features
#             self.classifier = None  # 只取特征

#     def construct(self, x):
#         if self.model_type == 'vgg16':
#             x = self.features(x)
#             x = x.view(x.shape[0], -1)
#             x = self.classifier(x)
#         elif self.model_type == 'mobilenet':
#             x = self.features(x)
#             x = ms.ops.adaptive_avg_pool2d(x, (1, 1))
#             x = x.view(x.shape[0], -1)
#         return x


# ✅ 对应 PyTorch 版本的 VGG16 特征提取
class VGG16Features(nn.Cell):
    def __init__(self):
        super(VGG16Features, self).__init__()
        vgg = vgg16(pretrained=True)
        self.features = vgg.features
        # 使用 VGG16 的分类器部分，但去掉最后一层
        original_classifier = vgg.classifier
        self.classifier = nn.SequentialCell([
            original_classifier[0],
            original_classifier[1],
            original_classifier[2],
            original_classifier[3],
            original_classifier[4],
            original_classifier[5]
        ])

    def construct(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


#手工特征提取部分（RGB,YcbCr,Lab,HSV）
def extract_features2(image):
    # 读取图像
    img = cv2.resize(image, (224, 224))  # 调整大小以保持一致

    # 计算 RGB 均值和标准差
    mean_rgb = np.mean(img, axis=(0, 1))  
    std_rgb = np.std(img, axis=(0, 1))  

    # 转换到 HSV 颜色空间
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_hsv = np.mean(img_hsv, axis=(0, 1))
    std_hsv = np.std(img_hsv, axis=(0, 1))

    # 转换到 Lab 颜色空间
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    mean_lab = np.mean(img_lab, axis=(0, 1))
    std_lab = np.std(img_lab, axis=(0, 1))

    # 转换到 YCbCr 颜色空间
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    mean_ycbcr = np.mean(img_ycbcr, axis=(0, 1))
    std_ycbcr = np.std(img_ycbcr, axis=(0, 1))

    #GLCM融合共生灰度矩阵
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 计算灰度共生矩阵（GLCM），设置不同的距离和方向
    glcm = graycomatrix(gray_image, 
                        distances=[1, 2, 3],  # 距离，可以选择1、2、3等不同值
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],  # 方向，0度、45度、90度、135度
                        symmetric=True, 
                        normed=True)
    
    # 从 GLCM 中提取特征
    contrast = graycoprops(glcm, 'contrast')  # 对比度
    energy = graycoprops(glcm, 'energy')  # 能量
    entropy = -np.sum(glcm * np.log2(glcm + 1e-6), axis=(0,1))  # 熵，避免log(0)的情况
    idm = np.sum([glcm[i,j] / (1 + (i-j)**2) for i in range(glcm.shape[0]) for j in range(glcm.shape[1])])
    
    # 将不同距离和方向的特征计算出来并合并成一个特征向量
    glcm_features = np.hstack([contrast.mean(axis=(0,1)),
                            energy.mean(axis=(0,1)),
                            entropy.mean(),
                            idm.mean()])

    # 组合所有特征
    features = np.hstack([mean_rgb, std_rgb, mean_hsv, std_hsv, mean_lab, std_lab, mean_ycbcr, std_ycbcr,glcm_features])
    return features

# CNN特征提取函数 - 对应PyTorch版本
def extract_features_mindspore(image, model, transform_pipeline):
    """使用MindSpore模型提取CNN特征"""
    # OpenCV图像转PIL图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = np.array(image)
    # 应用预处理
    data_transforms1 = transforms.Compose(transform_pipeline)
    image = data_transforms1(image)

    image = Tensor(image, dtype=mstype.float32)
    # 增加batch维度
    image = ms.ops.expand_dims(image, 0)

    # 提取特征
    model.set_train(False)
    features = model(image)
    features = ms.ops.flatten(features)

    return features.asnumpy()

# 添加Autoencoder特征降维函数 - 对应PyTorch版本
def compress_features_with_autoencoder(features, autoencoder_model, ckpt_path='D:/TCMinspection/autoencoder_model_mindspore.ckpt'):
    """使用MindSpore自编码器降维"""
    try:
        # 转换为MindSpore张量
        features_tensor = Tensor(features.astype(np.float32), mstype.float32)

        # 添加batch维度
        if len(features_tensor.shape) == 1:
            features_tensor = ms.ops.expand_dims(features_tensor, 0)

        # 使用自编码器降维
        compressed_features = encode_features(autoencoder_model, features_tensor, ckpt_path=ckpt_path)
        return compressed_features
    except Exception as e:
        print(f"Autoencoder降维失败: {e}")
        return features  # 返回原始特征

# 主处理流程 - 对应PyTorch版本优化
def main():
    # 初始化模型
    print("正在加载VGG16模型...")
    vgg_model = VGG16Features()
    vgg_model.set_train(False)

    # 初始化自编码器用于降维 (对应PyTorch版本功能)
    print("正在加载自编码器模型...")
    try:
        autoencoder_model = SupervisedAutoencoder(num_classes=5)
        autoencoder_ckpt = 'D:/TCMinspection/autoencoder_model_mindspore.ckpt'
    except Exception as e:
        print(f"自编码器初始化失败: {e}")
        autoencoder_model = None
        autoencoder_ckpt = None

    # 创建数据预处理管道
    data_transforms1 = create_data_transforms1()

    # 初始化面部检测器和关键点检测器
    print("正在加载面部检测模型...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("D:/dlib-predictor/shape_predictor_68_face_landmarks.dat")

    # 存储特征和标签 - 对应PyTorch版本
    encoder_list = []  # CNN特征
    feature_list = []  # 降维特征 + 手工特征融合
    label_list = []
    data_dir = 'D:/TCMinspection/imagesource/faces_original'

    print("开始处理图像...")

    # 处理所有图像
    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(data_dir, filename)
            print(f"处理图像: {filename}")
            # 提取标签特征
            label = filename.split('_')[1].split('.')[0]
            label_list.append(label)

            image = cv2.imread(img_path)
            if image is None:
                print(f"无法加载图像：{img_path}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print(gray.dtype, gray.shape)
            print(type(gray), gray.ndim)
            print("is contiguous:", gray.flags['C_CONTIGUOUS'])
            # 检测面部
            faces = detector(gray)
            print(gray.dtype, gray.shape)

            if len(faces) > 0:
                face = faces[0]
                landmarks = predictor(gray, face)
            else:
                print(f"未检测到面部：{filename}")
                continue

            # 获取关键点坐标
            ting_start = landmarks.part(21).x, landmarks.part(21).y  # 阙起点
            ting_end = landmarks.part(22).x, landmarks.part(22).y  # 阙终点
            mouth_left = landmarks.part(48).x, landmarks.part(48).y  # 口角左
            mouth_right = landmarks.part(54).x, landmarks.part(54).y  # 口角右
            chin_tip = landmarks.part(67).x, landmarks.part(67).y  # 下巴尖

            region_features1 = []  # CNN特征
            region_features2 = []  # 手工特征

            # 保存关键区域（阙，口角，下巴等共9个）
            for region_id in range(9):
                if region_id == 0:
                    region_image = image[mouth_left[1]-10:mouth_right[1]+15, mouth_left[0]-10:mouth_right[0]+15]
                elif region_id == 1:
                    region_image = image[ting_start[1]-10:ting_end[1]+10, ting_start[0]:ting_end[0]]
                elif region_id == 2:
                    region_image = image[landmarks.part(19).y-20:landmarks.part(19).y-10, landmarks.part(19).x+5:landmarks.part(19).x+100]
                elif region_id == 3:
                    region_image = image[landmarks.part(1).y-10:landmarks.part(1).y+10, landmarks.part(1).x+5:landmarks.part(1).x+25]
                elif region_id == 4:
                    region_image = image[landmarks.part(15).y-10:landmarks.part(15).y+10, landmarks.part(15).x-25:landmarks.part(15).x-5]
                elif region_id == 5:
                    region_image = image[landmarks.part(1).y+10:landmarks.part(1).y+40, landmarks.part(1).x+25:landmarks.part(1).x+50]
                elif region_id == 6:
                    region_image = image[landmarks.part(15).y+10:landmarks.part(15).y+40, landmarks.part(15).x-50:landmarks.part(15).x-25]
                elif region_id == 7:
                    region_image = image[landmarks.part(67).y+20:landmarks.part(67).y+40, landmarks.part(67).x-5:landmarks.part(67).x+25]
                elif region_id == 8:
                    region_image = image[landmarks.part(30).y-10:landmarks.part(30).y+10, landmarks.part(30).x-10:landmarks.part(30).x+10]

                if region_image is None or region_image.size == 0:
                    print(f"区域图像为空：{filename}, 区域：{region_id}")
                    continue
                else:
                    # 提取CNN特征 (对应PyTorch版本的features1)
                    features1 = extract_features_mindspore(region_image, vgg_model, data_transforms1)
                    encoder_list.append(features1)  # 保存原始CNN特征

                    # features_tensor = torch.from_numpy(features1).float().unsqueeze(0)
                    compressed_features = compress_features_with_autoencoder(features1, autoencoder_model, autoencoder_ckpt)
                    compressed_features = np.asarray(compressed_features).reshape(-1)  # 确保是一维数组
                    print(f"降维后特征 shape: {compressed_features.shape}")


                # 提取手工特征 (对应PyTorch版本的features2)
                features2 = extract_features2(region_image)
                print(f"手工特征 shape: {features2.shape}")

                # 存储特征 - 对应PyTorch版本
                region_features1.append(compressed_features)  # 降维后的CNN特征
                region_features2.append(features2)  # 手工特征

            # 将同一张图片的所有区域特征拼接 - 对应PyTorch版本

            if region_features1:
                final_features = np.hstack(region_features1 + region_features2)  # CNN+手工特征
                feature_list.append(final_features)
                print(f"最终特征 shape: {final_features.shape}")
            else:
                print(f"未能提取特征：{filename}")

        else:
            print(f"未检测到面部：{filename}")

    # 保存特征和标签 - 对应两个版本
    if feature_list:
        feature_array = np.array(feature_list)      # CNN+手工特征 (对应PyTorch版本)
        label_array = np.array(label_list)

        np.save('D:/TCMinspection/feature_mindspore_class.npy', feature_array)  # 保存融合特征
        np.save("D:/TCMinspection/label_mindspore_class.npy", label_array)

        print("特征提取完成！")
        print(f"提取的特征数量：{len(feature_list)}")
        print(f"融合特征维度：{feature_array.shape}")
    else:
        print("未能提取任何特征！")

if __name__ == "__main__":
    main()



