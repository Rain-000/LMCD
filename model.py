import torch
import torch.nn.functional as F
import torch.nn as nn


# 设置设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=100, out_channels=256, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=0.2)

        self.fc_input_size = 128 * (200 // 4)
        self.fc1 = nn.Linear(in_features=self.fc_input_size, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.out = nn.Linear(in_features=256, out_features=730)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.out(x)
        return x


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_size, num_layers):
        super(GRUModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers, bidirectional=True)

        self.dropout = nn.Dropout(p=0.05)
        self.fc = nn.Linear(hidden_dim * 2, out_size)

    def forward(self, features, hidden):
        output, hidden = self.gru(features, hidden)
        output = self.dropout(output)
        output = self.fc(output.view(-1, self.hidden_dim * 2))
        return output, hidden

    def init_zero_state(self, batch_size):
        return torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(DEVICE)


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(184, 120)
        self.linear2 = nn.Linear(120, 60)
        self.linear3 = nn.Linear(60, 30)
        self.linear4 = nn.Linear(30, 2)
        self.dropout = nn.Dropout(p=0.1)
        self.mish = Mish()

    def forward(self, x):
        out1 = self.dropout(self.mish(self.linear1(x)))
        out2 = self.dropout(self.mish(self.linear2(out1)))
        out3 = self.dropout(self.mish(self.linear3(out2)))
        return self.linear4(out3)


class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.cnn_model = CNNModel()
        self.gru_model = GRUModel(input_dim=730, hidden_dim=365, out_size=184, num_layers=2)

        # 自注意力层
        self.attention = nn.MultiheadAttention(embed_dim=184, num_heads=4, dropout=0.05)

        # 全局池化层
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # 线性层
        self.fc_global = nn.Linear(184 * 2, 184)
        self.linear_model = LinearModel()

    def forward(self, x):
        # CNN前向传播
        x = self.cnn_model(x)

        batch_size = x.size(0)
        hidden_state = self.gru_model.init_zero_state(batch_size)

        # GRU前向传播
        gru_input = x.unsqueeze(0)  # (1, batch_size, input_dim)
        gru_output, hidden_state = self.gru_model(gru_input, hidden_state)

        # 自注意力前向传播
        attn_input = gru_output.unsqueeze(0)  # (1, sequence_length, embed_dim)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)

        # 计算全局特征
        global_avg = self.global_avg_pool(attn_output.permute(1, 2, 0)).squeeze(-1)  # (batch_size, embed_dim)
        global_max = self.global_max_pool(attn_output.permute(1, 2, 0)).squeeze(-1)  # (batch_size, embed_dim)

        # 将全局特征与自注意力输出拼接
        global_feature = torch.cat((global_avg, global_max), dim=-1)  # (batch_size, embed_dim * 2)

        # 将拼接后的特征映射回embed_dim
        global_feature = self.fc_global(global_feature)

        # 使用拼接的全局特征和自注意力的输出进行线性模型的前向传播
        final_output = self.linear_model(global_feature)
        return final_output





# # 创建组合模型并移动到设备
# combined_model = CombinedModel().to(DEVICE)
#
# # 加载数据
# X_test_loaded = np.load('X_test.npy')
# x = X_test_loaded[0:32, :, :]
# X_train_tensor = torch.tensor(x, dtype=torch.float32).to(DEVICE)
# X_tensor = X_train_tensor.permute(0, 2, 1)
#
# # 使用组合模型进行前向传播
# final_output = combined_model(X_tensor)
# print("Final Output shape:", final_output.shape)
# print(final_output)
