import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.train import Model, LossMonitor
from mindspore.common import dtype as mstype
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 设置运行环境
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 加载特征和标签
data_np = np.load("D:/TCMinspection/feature_mindspore.npy")  # CNN提取的高维特征 (2700, 4096)
labels_np = np.load("D:/TCMinspection/label_mindspore.npy")   # 标签 (2700, )
data_np = np.squeeze(data_np, axis=1)  # 去掉多余的维度
labels_np = labels_np[:-1]
print(data_np.shape)
# print(np.unique(labels_np))
print(labels_np[:10])


# 数据预处理
labels_np = np.repeat(labels_np, 9)
le = LabelEncoder()
labels_encoded = le.fit_transform(labels_np)

print(labels_np.shape)

print(le.classes_)

# 转换为MindSpore张量
data = Tensor(data_np.astype(np.float32), mstype.float32)
labels = Tensor(labels_encoded.astype(np.int32), mstype.int32)

# 2. 定义监督自编码器
class SupervisedAutoencoder(nn.Cell):
    def __init__(self, num_classes=5):
        super(SupervisedAutoencoder, self).__init__()
        self.encoder = nn.SequentialCell([
            nn.Dense(4096, 1024),
            nn.ReLU(),
            nn.Dense(1024, 256),
            nn.ReLU(),
            nn.Dense(256, 128)
        ])
        self.decoder = nn.SequentialCell([
            nn.Dense(128, 256),
            nn.ReLU(),
            nn.Dense(256, 1024),
            nn.ReLU(),
            nn.Dense(1024, 4096)
        ])
        self.classifier = nn.Dense(128, num_classes)

    #     # 权重初始化 - 修复梯度消失问题
    #     self._initialize_weights()

    # def _initialize_weights(self):
    #     """Xavier权重初始化"""
    #     for cell in self.encoder:
    #         if isinstance(cell, nn.Dense):
    #             cell.weight.set_data(ms.common.initializer.initializer(
    #                 ms.common.initializer.XavierUniform(), cell.weight.shape, cell.weight.dtype))
    #     for cell in self.decoder:
    #         if isinstance(cell, nn.Dense):
    #             cell.weight.set_data(ms.common.initializer.initializer(
    #                 ms.common.initializer.XavierUniform(), cell.weight.shape, cell.weight.dtype))
    #     self.classifier.weight.set_data(ms.common.initializer.initializer(
    #         ms.common.initializer.XavierUniform(), self.classifier.weight.shape, self.classifier.weight.dtype))

    def construct(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        logits = self.classifier(encoded)
        return decoded, logits

# 3. 定义损失函数
class SupervisedAutoencoderLoss(nn.Cell):
    def __init__(self, alpha=1.0, beta=0.5):
        super(SupervisedAutoencoderLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def construct(self, decoded, original, logits, labels):
        reconstruction_loss = self.mse_loss(decoded, original)
        classification_loss = self.ce_loss(logits, labels)
        # 重要：不要对损失函数重复求均值，保持梯度一致性
        total_loss = self.alpha * reconstruction_loss + self.beta * classification_loss
        return total_loss

# 4. 训练函数
def train_supervised_autoencoder(model, data, labels, epochs=100, batch_size=64, lr=1e-3, alpha=0.5, beta=0.5):
    # 创建数据集
    dataset = ms.dataset.NumpySlicesDataset({"data": data.asnumpy(), "labels": labels.asnumpy()}, shuffle=True)
    dataset = dataset.batch(batch_size)

    # 定义优化器
    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)

    # 定义损失函数
    loss_fn = SupervisedAutoencoderLoss(alpha=alpha, beta=beta)

    # 训练模型 - 修复关键训练逻辑
    print("开始训练监督自编码器...")
    def forward_fn(batch_x, batch_y):
        decoded, logits = model(batch_x)
        loss = loss_fn(decoded, batch_x, logits, batch_y)
        return loss, decoded, logits

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    for epoch in range(epochs):
        model.set_train()
        total_loss = 0.0
        step = 0

        for batch_data in dataset.create_dict_iterator():
            batch_x = Tensor(batch_data["data"], mstype.float32)
            batch_y = Tensor(batch_data["labels"], mstype.int32)

            # 前向传播 + 反向传播 + 参数更新
            (loss, _, _), grads = grad_fn(batch_x, batch_y)
            optimizer(grads)

            total_loss += loss.asnumpy()
            step += 1

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/step:.4f}")

# 5. 编码器特征提取函数
def encode_features(model, data, ckpt_path="autoencoder_model_mindspore.ckpt"):
    # 加载模型参数
    param_dict = ms.load_checkpoint(ckpt_path)
    ms.load_param_into_net(model, param_dict)
    model.set_train(False)

    encoded_features = []
    for i in range(0, data.shape[0], 64):  # 批量处理
        batch_data = data[i:i+64]
        encoded = model.encoder(batch_data)
        encoded_features.append(encoded.asnumpy())

    return np.vstack(encoded_features)

# 6. 主流程
if __name__ == "__main__":
    num_classes = len(np.unique(labels_np))  # 自动根据标签数量确定分类数
    print(num_classes)
    model = SupervisedAutoencoder(num_classes=num_classes)

    print("开始训练监督自编码器...")
    train_supervised_autoencoder(model, data, labels, epochs=100, batch_size=64, lr=1e-3, alpha=0.3, beta=0.7)

    print("保存训练好的模型...")
    ms.save_checkpoint(model, "autoencoder_model_mindspore.ckpt")

    print("全部完成！")