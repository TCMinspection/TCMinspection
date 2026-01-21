import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 加载特征和标签
data_np = np.load("D:/TCMinspection/newencoder.npy")  # CNN提取的高维特征 (2700, 4096)
labels_np = np.load("D:/TCMinspection/label_mindspore.npy")   # 标签 (2700, )
print(data_np.shape)
print(np.unique(labels_np))
# 转为 torch tensor
data = torch.tensor(data_np, dtype=torch.float32)
labels_np=np.repeat(labels_np,9)
le = LabelEncoder()
labels_encoded = le.fit_transform(labels_np) 

print(le.classes_)
labels = torch.tensor(labels_encoded, dtype=torch.long)

# 2. 定义监督自编码器
class SupervisedAutoencoder(nn.Module):
    def __init__(self, num_classes=5):
        super(SupervisedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        logits = self.classifier(encoded)
        return decoded, logits

#3. 训练函数
def train_supervised_autoencoder(model, data, labels, epochs=100, batch_size=64, lr=1e-3, alpha=1.0, beta=0.5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    reconstruction_loss_fn = nn.MSELoss()
    classification_loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(data.size(0))
        total_loss = 0.0
        for i in range(0, data.size(0), batch_size):
            batch_data = data[perm[i:i+batch_size]]
            batch_labels = labels[perm[i:i+batch_size]]

            decoded, logits = model(batch_data)
            reconstruction_loss = reconstruction_loss_fn(decoded, batch_data)
            classification_loss = classification_loss_fn(logits, batch_labels)

            loss = alpha * reconstruction_loss + beta * classification_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

# 4. 编码器特征提取函数
def encode_features(model, data, ckpt_path="autoencoder_model.pth"):
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    with torch.no_grad():
        encoded_features = model.encoder(data)
    return encoded_features.cpu().numpy()  # 返回 numpy

#5. 主流程
if __name__ == "__main__":
    num_classes = len(np.unique(labels_np))  # 自动根据标签数量确定分类数
    print(num_classes)
    model = SupervisedAutoencoder(num_classes=num_classes)

    print("开始训练监督自编码器...")
    train_supervised_autoencoder(model, data, labels, epochs=100, batch_size=64, lr=1e-3, alpha=1.0, beta=0.5)

    print("保存训练好的模型...")
    torch.save(model.state_dict(), "autoencoder_model.pth")

    print("全部完成！")
