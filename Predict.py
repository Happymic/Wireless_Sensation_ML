# 测试数据集
from TrainDataset import TrainDataset
from TestDataset import TestDataset
from Train import collate_fn
from Train import model
from torch.utils.data import DataLoader
import torch
print("Loading: ")
test_data = TestDataset()
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_fn)
print("Loading: ")
model.eval()
correct = 0
total = 0
predicted_labels = []
print("Start")
with torch.no_grad():
    for rssi, agc, csi_real, csi_imag, labels in test_loader:
        rssi = torch.stack(rssi).double()
        agc = torch.stack(agc).double()
        csi_real = torch.stack(csi_real).double()
        csi_imag = torch.stack(csi_imag).double()
        inputs = torch.cat((rssi, agc, csi_real, csi_imag), dim=-1)

        outputs = model(inputs)


        _, predicted = torch.max(outputs, 1)
        predicted_labels.extend(predicted.cpu().numpy())

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


#print(f'测试精度: {100 * correct / total:.2f}%')
print("预测的房间人数:", predicted_labels)
