from flask import Flask, request, render_template, jsonify
import cv2
import dlib
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from flask_cors import CORS  # 导入 CORS
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)  # 配置 CORS，允许所有来源的跨域请求

# 数据预处理
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 导入模型
model = models.vgg16(pretrained=True)
model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])
model.eval()

# 定义特征提取函数
def extract_features1(image_path, model, data_transforms):
    image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = data_transforms(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    features = torch.flatten(features)
    return features.numpy()

def extract_features2(image):
    img = cv2.resize(image, (224, 224))
    mean_rgb = np.mean(img, axis=(0, 1))
    std_rgb = np.std(img, axis=(0, 1))

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_hsv = np.mean(img_hsv, axis=(0, 1))
    std_hsv = np.std(img_hsv, axis=(0, 1))

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    mean_lab = np.mean(img_lab, axis=(0, 1))
    std_lab = np.std(img_lab, axis=(0, 1))

    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    mean_ycbcr = np.mean(img_ycbcr, axis=(0, 1))
    std_ycbcr = np.std(img_ycbcr, axis=(0, 1))
    
    features = np.hstack([mean_rgb, std_rgb, mean_hsv, std_hsv, mean_lab, std_lab, mean_ycbcr, std_ycbcr])
    return features

# 初始化面部检测器和关键点检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/dlib-predictor/shape_predictor_68_face_landmarks.dat")

def extract_image_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    face = faces[0]
    landmarks = predictor(gray, face)

    ting_start = landmarks.part(21).x, landmarks.part(21).y
    ting_end = landmarks.part(22).x, landmarks.part(22).y
    mouth_left = landmarks.part(48).x, landmarks.part(48).y
    mouth_right = landmarks.part(54).x, landmarks.part(54).y
    chin_tip = landmarks.part(67).x, landmarks.part(67).y

    region_features = []
    for region_id in range(9):
        if region_id == 0:
            region_image = image[mouth_left[1] - 10:mouth_right[1] + 15, mouth_left[0] - 10:mouth_right[0] + 15]
        elif region_id == 1:
            region_image = image[ting_start[1] - 10:ting_end[1] + 10, ting_start[0]:ting_end[0]]
        elif region_id == 2:
            region_image = image[landmarks.part(19).y - 20:landmarks.part(19).y - 10,
                                landmarks.part(19).x + 5:landmarks.part(19).x + 100]
        elif region_id == 3:
            region_image = image[landmarks.part(1).y - 10:landmarks.part(1).y + 10,
                                landmarks.part(1).x + 5:landmarks.part(1).x + 25]
        elif region_id == 4:
            region_image = image[landmarks.part(15).y - 10:landmarks.part(15).y + 10,
                                landmarks.part(15).x - 25:landmarks.part(15).x - 5]
        elif region_id == 5:
            region_image = image[landmarks.part(1).y + 10:landmarks.part(1).y + 40,
                                landmarks.part(1).x + 25:landmarks.part(1).x + 50]
        elif region_id == 6:
            region_image = image[landmarks.part(15).y + 10:landmarks.part(15).y + 40,
                                landmarks.part(15).x - 50:landmarks.part(15).x - 25]
        elif region_id == 7:
            region_image = image[landmarks.part(67).y + 20:landmarks.part(67).y + 40,
                                landmarks.part(67).x - 5:landmarks.part(67).x + 25]
        elif region_id == 8:
            region_image = image[landmarks.part(30).y - 10:landmarks.part(30).y + 10,
                                landmarks.part(30).x - 10:landmarks.part(30).x + 10]
        features1 = extract_features1(region_image, model, data_transforms)
        features2 = extract_features2(region_image)
        combined_features = np.concatenate((features1, features2))
        region_features.append(combined_features)

    final_features = np.concatenate(region_features)
    print(final_features.shape)
    return final_features

svm_model = joblib.load("D:/Facerecognition/svm_model.pkl")
scaler = joblib.load("D:/Facerecognition/scaler.pkl")

@app.route('/diagnose', methods=['POST'])
def diagnose():
    file = request.files['image']
    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        features = extract_image_features(img)
        if features is None:
            return jsonify({'error': '未检测到人脸'})
        features_scaled = scaler.transform([features])
        prediction = svm_model.predict(features_scaled)[0]
        return jsonify({'prediction': prediction})
    return jsonify({'error': '未提供图片'})

@app.route('/real-time-diagnose', methods=['POST'])
def real_time_diagnose():
    file = request.files['image']
    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        features = extract_image_features(img)
        if features is None:
            return jsonify({'error': '未检测到人脸'})
        features_scaled = scaler.transform([features])
        prediction = svm_model.predict(features_scaled)[0]
        return jsonify({'prediction': prediction})
    return jsonify({'error': '未提供图片'})

if __name__ == '__main__':
    app.run(debug=True)
    