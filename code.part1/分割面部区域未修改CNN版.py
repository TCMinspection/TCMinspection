
import cv2
import dlib
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from facenet_pytorch import InceptionResnetV1
from torchvision.models import mobilenet_v3_large
from skimage.feature import graycomatrix, graycoprops
from Autoencoder import SupervisedAutoencoder,encode_features
# import sys
# from insightface.model_zoo import get_model

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "D:/Facerecognition/arcface-pytorch-main/nets")))

# from arcface import ArcFace
# 数据预处理
data_transforms1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet的均值
                        std=[0.229, 0.224, 0.225])   # ImageNet的标准差
])

data_transforms2 =transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5])
])

data_transforms3 = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

#导入模型vgg16:0.51
model=models.vgg16(pretrained=True)
model.classifier=nn.Sequential(*list(model.classifier.children())[:-1])
model.eval()

#导入模型facenet0.45

facenet_model=InceptionResnetV1(pretrained='casia-webface').eval()

#导入模型mobilenet:0.49
mobilnet_model=mobilenet_v3_large(pretrain=True)
mobilenet_model = mobilnet_model.features
mobilnet_model.eval()

# arcface_model = ArcFace(backbone="mobilefacenet",mode="predict")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# arcface_model.load_state_dict(torch.load("D:/Facerecognition/Tools/arcface_mobilefacenet.pth", map_location=device), strict=False)
# arcface_model  = model.eval()

#导入自编码器
model2 = SupervisedAutoencoder()

#定义特征提取函数
#CNN提取的特征
def extract_features1(image_path,model,data_transforms):
    image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image =data_transforms(image)
    image=image.unsqueeze(0)
    with torch.no_grad():
        features=model(image)
    features=torch.flatten(features)
    return features.numpy()

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

#ArcFace提取的特征
# def extract_features3(image):
#     image=cv2.resize(image,(112,112))
#     image=Image.fromarray(image)
#     image=data_transforms2(image)
#     image=image.unsqueeze(0)
#     face = arcface_model.get(image)
#     return face[0]

#facenet提取特征
def extract_features4(image_path,model,data_transforms):
    image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image =data_transforms(image)
    image=image.unsqueeze(0)
    with torch.no_grad():
        features=model(image)
    features=torch.flatten(features)
    return features.numpy()
    

#Mobilenet提取特征
def extract_features5(image_path,model,data_transforms):
    image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image =data_transforms(image)
    image=image.unsqueeze(0)
    with torch.no_grad():
        features=model(image)
    features=torch.flatten(features)
    return features.numpy()

encoder_list=[]
feature_list=[]
label_list = []
data_dir='D:/TCMinspection/imagesource1'

# 初始化面部检测器和关键点检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/dlib-predictor/shape_predictor_68_face_landmarks.dat")

# 加载图像
# image_path ='D:/Facerecognition/imagesource1/picture1.jpg'
# image = cv2.imread(image_path)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 检测面部
for filename in os.listdir(data_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(data_dir, filename)

        #标签特征提取
        label = filename.split('_')[1].split('.')[0]
        label_list.append(label)

        image = cv2.imread(img_path)
        if image is None:
            print(f"无法加载图像：{img_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 检测面部
        faces = detector(gray)
        if len(faces) > 0:
            face = faces[0]
            landmarks = predictor(gray, face)
        else:
            print(filename)
        # 获取关键点坐标
        ting_start = landmarks.part(21).x, landmarks.part(21).y  # 阙起点
        ting_end = landmarks.part(22).x, landmarks.part(22).y  # 阙终点
        mouth_left = landmarks.part(48).x, landmarks.part(48).y  # 口角左
        mouth_right = landmarks.part(54).x, landmarks.part(54).y  # 口角右
        chin_tip = landmarks.part(67).x, landmarks.part(67).y  # 下巴尖

    # 绘制关键区域的矩形框
    # cv2.rectangle(image, (ting_start[0], ting_start[1] - 10), (ting_end[0], ting_end[1] + 10), (0, 255, 0), 2)#阙  finish
    # cv2.rectangle(image,(landmarks.part(19).x+5,landmarks.part(19).y-10),(landmarks.part(19).x+100,landmarks.part(19).y-30),(0,225,0),2)#庭  finish
    # cv2.rectangle(image, (mouth_left[0] - 10, mouth_left[1] - 10), (mouth_right[0] + 15, mouth_right[1] + 15), (0, 255, 0), 2)#嘴       finish
    # cv2.rectangle(image,(landmarks.part(1).x+5,landmarks.part(1).y-10),(landmarks.part(1).x+25,landmarks.part(1).y+10),(0,225,0),2)#左颧骨   finish
    # cv2.rectangle(image,(landmarks.part(15).x-5,landmarks.part(15).y-10),(landmarks.part(15).x-25,landmarks.part(15).y+10),(0,225,0),2)#右颧骨 finish
    # cv2.rectangle(image,(landmarks.part(1).x+25,landmarks.part(1).y+10),(landmarks.part(1).x+50,landmarks.part(1).y+40),(0,225,0),2)#左脸颊    finish
    # cv2.rectangle(image,(landmarks.part(15).x-25,landmarks.part(15).y+10),(landmarks.part(15).x-50,landmarks.part(15).y+40),(0,225,0),2)#右脸颊 finish
    # cv2.rectangle(image,(landmarks.part(67).x-5,landmarks.part(67).y+40),(landmarks.part(67).x+25,landmarks.part(67).y+20),(0,225,0),2)#下巴（颏）
    # cv2.rectangle(image,(landmarks.part(30).x-10,landmarks.part(30).y-10),(landmarks.part(30).x+10,landmarks.part(30).y+10),(0,225,0),2)#明堂 finish

    # print((landmarks.part(19).x+5,landmarks.part(19).y-10),(landmarks.part(19).x+100,landmarks.part(19).y-30))#庭
    # print((landmarks.part(15).x-5,landmarks.part(15).y-10),(landmarks.part(15).x-25,landmarks.part(15).y+10))#右颧骨
    # print((landmarks.part(15).x-25,landmarks.part(15).y+10),(landmarks.part(15).x-50,landmarks.part(15).y+40))#右脸夹
    # print((landmarks.part(67).x-5,landmarks.part(67).y+40),(landmarks.part(67).x+25,landmarks.part(67).y+20))#下巴
        region_features1=[]
        region_features2=[]
        #保存关键区域（阙，口角，下巴等共9个）
        for region_id in range(9):
            if region_id==0:
                region_image=image[mouth_left[1]-10:mouth_right[1]+15,mouth_left[0]-10:mouth_right[0]+15]   
            elif region_id==1:
                region_image=image[ting_start[1]-10:ting_end[1]+10,ting_start[0]:ting_end[0]]
            elif region_id==2:
                region_image=image[landmarks.part(19).y-20:landmarks.part(19).y-10,landmarks.part(19).x+5:landmarks.part(19).x+100]
            elif region_id==3:
                region_image=image[landmarks.part(1).y-10:landmarks.part(1).y+10,landmarks.part(1).x+5:landmarks.part(1).x+25]    
            elif region_id==4:
                region_image=image[landmarks.part(15).y-10:landmarks.part(15).y+10,landmarks.part(15).x-25:landmarks.part(15).x-5]
            elif region_id==5:
                region_image=image[landmarks.part(1).y+10:landmarks.part(1).y+40,landmarks.part(1).x+25:landmarks.part(1).x+50]
            elif region_id==6:
                region_image=image[landmarks.part(15).y+10:landmarks.part(15).y+40,landmarks.part(15).x-50:landmarks.part(15).x-25]
            elif region_id==7:
                region_image=image[landmarks.part(67).y+20:landmarks.part(67).y+40,landmarks.part(67).x-5:landmarks.part(67).x+25]
            elif region_id==8:
                region_image=image[landmarks.part(30).y-10:landmarks.part(30).y+10,landmarks.part(30).x-10:landmarks.part(30).x+10]

            if region_image is None or region_image.size == 0:
                print(filename)
            else:
                # features2 = extract_features2(region_image)
                features1 = extract_features1(region_image,model,data_transforms1)
                encoder_list.append(features1)
                with torch.no_grad():
                    features1=torch.tensor(features1,dtype=torch.float32)
                    compressed_features = encode_features(model2,features1,ckpt_path='D:/TCMinspection/autoencoder_model.pth') 

                    print("降维后特征 shape:", compressed_features.shape)
                features2=extract_features2(region_image)
                print(features2.shape)

            # 存储，标准化，特征
            region_features1.append(compressed_features)
            region_features2.append(features2)

        # 将同一张图片 的所有区域特征拼接
        encoder_features=np.hstack(region_features1)
        final_features = np.hstack(region_features1+region_features2)
        print(final_features.shape)
        # encoder_list.extend(region_features1)
        feature_list.append(final_features)
    else:
        print("No face detected")

encoder_array=np.array(encoder_list)
feature_array=np.array(feature_list)
label_array = np.array(label_list)

# np.save('D:/TCMinspection/newencoder.npy',encoder_array)
np.save("D:/TCMinspection/feature.npy", feature_array)
np.save("D:/TCMinspection/label.npy", label_array)

print("特征提取完成！")
print("特征向量：",region_features1)
print("标签",label_list)



