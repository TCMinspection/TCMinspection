import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# 1. 加载特征和标签
combined_features = np.load('D:/TCMinspection/feature_mindspore_class.npy')
labels = np.load('D:/TCMinspection/label_mindspore_class.npy')
print("特征 shape:", combined_features.shape)

# 假设特征前 1152 维是 CNN（自编码器）特征，后 252 维是手工特征
cnn_dim = 1152
hand_dim = 252
# assert combined_features.shape[1] == cnn_dim + hand_dim, "维度不一致，请检查！"

# combined_features = combined_features[:-1]
labels = labels[:-1]

print("对齐后的数据维度：", combined_features.shape, labels.shape)
# 2. 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    combined_features, labels, test_size=0.13,random_state=45)

# 3. 创建 ColumnTransformer：分别标准化 CNN 特征和手工特征
ct = ColumnTransformer([
    ('scale_cnn', StandardScaler(), slice(0, cnn_dim)),
    ('scale_hand', StandardScaler(), slice(cnn_dim, cnn_dim + hand_dim))
])

# 4. 创建 Pipeline：先分别标准化 → 再交给 SVM 分类器
pipe = Pipeline([
    ('scaler', ct),
    ('svm', SVC(kernel='linear', C=10.0, gamma='scale'))
])

# 5. 拟合模型
pipe.fit(X_train, y_train)

# 6. 预测与评估
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Pipeline SVM 准确率:", acc)

# 7. 保存整个 pipeline（包含 scaler + svm）
joblib.dump(pipe, 'svm_pipeline.pkl')
