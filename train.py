import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from LOSS import Loss
from model import CombinedModel
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

# 加载数据
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# 检查数据类型并转换为 float
X_train = X_train.astype(np.float32)  # 将 X_train 转换为 float32
y_train = y_train.astype(np.int64)     # 将 y_train 转换为 int64（对于分类标签）

X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.int64)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

model = CombinedModel().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# 训练循环
num_epochs = 20
for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    total_loss = 0

    # 使用 tqdm 包裹 train_loader 以显示进度条
    with tqdm(train_loader, unit='batch') as tepoch:
        for x, y in tepoch:
            x, y = x.to(DEVICE), y.to(DEVICE)  # 移动到设备

            # permute 使输入的维度变为 (batch_size, 100, 300)
            x = x.permute(0, 2, 1)  # 调整输入的维度
            optimizer.zero_grad()  # 清零梯度
            y_pred = model(x)  # 前向传播
            loss_function = Loss().to(DEVICE)
            loss = loss_function.get_loss(y_pred, y) # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数


            total_loss += loss.item()  # 累加损失

            # 更新进度条描述
            tepoch.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
            tepoch.set_postfix(loss=total_loss / (tepoch.n + 1))  # 显示当前批次的平均损失

    # 计算每个 epoch 的平均损失
    avg_loss = total_loss / len(train_loader)
    scheduler.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    from sklearn.metrics import f1_score

    # 在每个 epoch 结束时评估测试集
    model.eval()  # 设置为评估模式
    total_test_loss = 0
    correct = 0
    total = 0
    all_preds = []  # 用于存储所有预测
    all_labels = []  # 用于存储所有真实标签

    with torch.no_grad():  # 禁用梯度计算
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)  # 移动到设备
            x = x.permute(0, 2, 1)  # 调整输入的维度
            y_pred = model(x)  # 前向传播

            loss = loss_function.get_loss(y_pred, y)  # 计算损失
            total_test_loss += loss.item()  # 累加测试损失

            # 计算准确率
            _, predicted = torch.max(y_pred, 1)  # 获取预测结果
            total += y.size(0)  # 增加总样本数
            correct += (predicted == y).sum().item()  # 计算正确预测的数量

            # 存储预测和真实标签
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = correct / total * 100  # 计算准确率

    # 计算 F1 值
    f1 = f1_score(all_labels, all_preds, average='weighted')  # 使用加权平均

    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, F1 Score: {f1:.4f}')

torch.save(model.state_dict(), './Model1.pth')