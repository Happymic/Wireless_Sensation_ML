import torch
from TrainDataset import TrainDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# 定义 RNN 模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# 定义超参数
input_size = 2000
hidden_size = 64
num_classes = 4
num_epochs = 5


def collate_fn(batch):
    rssi_list, agc_list, csi_real_list, csi_imag_list, labels_list = zip(*batch)

    # 找到 rssi_list 和 agc_list 的最大长度，并确保它们的长度不小于（新的数据格式我还没看明白）
    max_length = max(256, max(max([len(rssi) for rssi in rssi_list]), max([len(agc) for agc in agc_list])))

    # 对 rssi_list 和 agc_list 进行填充，并确保它们的类型为 float64
    rssi_list = [torch.tensor(
        np.pad(rssi.astype(np.float64), ((0, 0), (0, max(0, max_length - rssi.shape[1]))), 'constant',
               constant_values=0))[:256, :max_length] for rssi in rssi_list]
    agc_list = [torch.tensor(
        np.pad(agc.astype(np.float64), ((0, 0), (0, max(0, max_length - agc.shape[1]))), 'constant',
               constant_values=0))[:256, :max_length] for agc in agc_list]

    # 找到 csi_real_list 和 csi_imag_list 的最大长度，并对它们进行填充
    max_length_csi = max(max([csi_real.shape[1] for csi_real in csi_real_list]), max([csi_imag.shape[1] for csi_imag in csi_imag_list]))
    csi_real_list = [torch.tensor(np.pad(csi_real, ((0, 0), (0, max_length - csi_real.shape[1])), 'constant', constant_values=0))[:256, :max_length] for csi_real in csi_real_list]
    csi_imag_list = [torch.tensor(np.pad(csi_imag, ((0, 0), (0, max_length - csi_imag.shape[1])), 'constant', constant_values=0))[:256, :max_length] for csi_imag in csi_imag_list]

    labels = torch.tensor(labels_list)

    return rssi_list, agc_list, csi_real_list, csi_imag_list, labels


train_data = TrainDataset()
train_loader = DataLoader(train_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

# # 创建 RNN 模型
# model = RNN(input_size, hidden_size, num_classes)

model = RNN(input_size, hidden_size, num_classes)
model = model.double()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
print(criterion)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(num_epochs):
    for rssi, agc, csi_real, csi_imag, labels in train_loader:
        rssi = torch.stack(rssi).double()
        agc = torch.stack(agc).double()
        csi_real = torch.stack(csi_real).double()  # 将 csi_real 列表转换为 double 类型的张量
        csi_imag = torch.stack(csi_imag).double()
        inputs = torch.cat((rssi, agc, csi_real, csi_imag), dim=-1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
print("------Done training-----")