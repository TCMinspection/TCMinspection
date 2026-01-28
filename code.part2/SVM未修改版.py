import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# 1. 加载特征和标签（使用相对路径，便于在仓库内运行）
combined_features = np.load('feature_mindspore_class.npy')
labels = np.load('label_mindspore_class.npy')
print("特征 shape:", combined_features.shape)

# 假设特征前 1152 维是 CNN（自编码器）特征，后 252 维是手工特征
cnn_dim = 1152
hand_dim = 252

labels = labels[:-1]

print("对齐后的数据维度：", combined_features.shape, labels.shape)

# 2. 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    combined_features, labels, test_size=0.13, random_state=45)

# 3. 创建 ColumnTransformer：分别标准化 CNN 特征和手工特征
ct = ColumnTransformer([
    ('scale_cnn', StandardScaler(), slice(0, cnn_dim)),
    ('scale_hand', StandardScaler(), slice(cnn_dim, cnn_dim + hand_dim))
])

# 4. 使用 MLPClassifier（MLP 相对容易适配多种特征，且支持 early_stopping）
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 128),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

pipe = Pipeline([
    ('scaler', ct),
    ('mlp', mlp)
])

# 5. 拟合模型
pipe.fit(X_train, y_train)

# 6. 预测与评估
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Pipeline MLP 准确率:", acc)

# 7. 保存整个 pipeline（包含 scaler + mlp）
joblib.dump(pipe, 'mlp_pipeline.pkl')
