import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, l2_lambda=0.01):
        super(Loss, self).__init__()
        self.l2_lambda = l2_lambda  # L2 正则化系数
        self.criterion = nn.CrossEntropyLoss()  # 交叉熵损失

    def forward(self, y_pred, y, model=None):
        # 计算交叉熵损失
        loss = self.criterion(y_pred, y)

        # 如果提供了模型，则计算 L2 正则化损失
        if model is not None:
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param) ** 2  # 计算所有参数的 L2 范数

            loss += self.l2_lambda * l2_reg  # 添加 L2 正则化项

        return loss

    def get_loss(self, y_pred, y, model=None):
        return self.forward(y_pred, y, model)  # 调用 forward 方法计算损失
