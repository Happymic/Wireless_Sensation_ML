from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class TrainDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.max_length = 500  # 根据您的数据集设置最大序列长度
        self.rssi_matrixs, self.ag_matrixs, self.csi_matrixs, self.targets = self._get_data()


    def __getitem__(self, index):
        csi_real = self.csi_matrixs[index].real
        csi_imag = self.csi_matrixs[index].imag
        return self.rssi_matrixs[index], self.ag_matrixs[index], csi_real.T, csi_imag.T, self.targets[index]

    def __len__(self):
        return len(self.targets)

    def _get_data(self):
        rssi_matrixs, ag_matrixs, csi_matrixs, targets = [], [], [], []
        csi_files = ['/Users/michaelli/Downloads/AP_Left/data/csi_2023_10_20_3.txt',
                    '/Users/michaelli/Downloads/AP_Left/data/csi_2023_10_20_6.txt']
        csi_files_1 = ['/Users/michaelli/Downloads/AP_Left/data/csi_2023_10_20_5.txt']
        csi_labels = ['/Users/michaelli/Downloads/AP_Left/truth/csi_2023_10_20_3_truth.txt',
                     '/Users/michaelli/Downloads/AP_Left/truth/csi_2023_10_20_6_truth.txt']
        csi_labels_1 = ['/Users/michaelli/Downloads/AP_Left/truth/csi_2023_10_20_5_truth.txt']

        print("正在加载数据")
        for i in tqdm(range(len(csi_files))):
            rssi_matrix, ag_matrix, csi_matrix, labels = self._get_matrix(
                csi_file=csi_files[i],
                csi_label=csi_labels[i]
            )
            # 对 rssi_matrix 和 ag_matrix 进行填充
            rssi_matrix = [np.pad(rssi, (0, self.max_length - len(rssi)), 'constant', constant_values=0) for rssi in
                           rssi_matrix]
            ag_matrix = [np.pad(ag, (0, self.max_length - len(ag)), 'constant', constant_values=0) for ag in ag_matrix]

            # 删除这两行，因为我们将在循环内部添加数据
            # rssi_matrixs.extend(rssi_matrix)
            # ag_matrixs.extend(ag_matrix)
            # csi_matrixs.extend(csi_matrix)
            # targets.extend(labels)

            for j in range(len(rssi_matrix)):
                if j < len(labels):
                    rssi_matrixs.append(rssi_matrix[j])
                    ag_matrixs.append(ag_matrix[j])
                    csi_matrixs.append(csi_matrix[j])
                    targets.append(labels[j])

        for i in tqdm(range(len(csi_files_1))):
            rssi_matrix, ag_matrix, csi_matrix, labels = self._get_matrix_1(
                csi_file_1=csi_files_1[i],
                csi_label=csi_labels_1[i]
            )
            # 对 rssi_matrix 和 ag_matrix 进行填充
            rssi_matrix = [np.pad(rssi, (0, self.max_length - len(rssi)), 'constant', constant_values=0) for rssi in
                           rssi_matrix]
            ag_matrix = [np.pad(ag, (0, self.max_length - len(ag)), 'constant', constant_values=0) for ag in ag_matrix]

            # 删除这两行，因为我们将在循环内部添加数据
            # rssi_matrixs.extend(rssi_matrix)
            # ag_matrixs.extend(ag_matrix)
            # csi_matrixs.extend(csi_matrix)
            # targets.extend(labels)

            for j in range(len(rssi_matrix)):
                if j < len(labels):
                    rssi_matrixs.append(rssi_matrix[j])
                    ag_matrixs.append(ag_matrix[j])
                    csi_matrixs.append(csi_matrix[j])
                    targets.append(labels[j])

        print("Loaded data:")
        print("rssi_matrixs:", len(rssi_matrixs))
        print("ag_matrixs:", len(ag_matrixs))
        print("csi_matrixs:", len(csi_matrixs))
        print("targets:", len(targets))

        return rssi_matrixs, ag_matrixs, csi_matrixs, targets


    def _get_matrix(self, csi_file, csi_label=None):
        with open(csi_file, 'r') as file:
            lines = file.readlines()
        rssi_matrix, ag_matrix, csi_matrix = [], [], []
        rssi_data, ag_data, csi_data = [], [], []
        timestamps, current_timestamp = [], 0
        count = 0  # 添加一个计数器
        for i, line in enumerate(lines):
            data = line.strip().split()
            h, m, s = map(float, data[:3])
            rssi = list(map(int, data[3:7]))
            ag = list(map(int, data[8:12]))
            csi = [complex(data[i].replace('i', 'j')) for i in range(12, len(data))]
            timestamp = h * 3600 + m * 60 + s
            timestamps.append(timestamp)
            if timestamp - current_timestamp < 2:
                rssi_data.append(rssi)
                ag_data.append(ag)
                csi_data.append(csi)
            else:
                rssi_matrix.append(np.array(rssi_data))
                ag_matrix.append(np.array(ag_data))
                csi_matrix.append(np.array(csi_data))
                count += 1  # 在添加数据到 rssi_matrix, ag_matrix, csi_matrix 时递增计数器
                rssi_data, ag_data, csi_data = [], [], []
                current_timestamp += 2

        if len(rssi_data):
            rssi_matrix.append(np.array(rssi_data))
            ag_matrix.append(np.array(ag_data))
            csi_matrix.append(np.array(csi_data))
            count += 1  # 在添加数据到 rssi_matrix, ag_matrix, csi_matrix 时递增计数器

        if csi_label:
            with open(csi_label, 'r', encoding='utf-8') as file:
                labels = [int(k) for k in file.read().split()]
        else:
            labels = [0 for _ in range(count)]  # 根据计数器的值创建具有相同长度的 labels 列表

        return rssi_matrix, ag_matrix, csi_matrix, labels

    def _get_matrix_1(self, csi_file_1, csi_label=None):
        with open(csi_file_1, 'r') as file:
            lines = file.readlines()

        rssi_matrix, ag_matrix, csi_matrix = [], [], []
        rssi_data, ag_data, csi_data = [], [], []
        timestamps, current_timestamp = [], 0
        fre_res = []

        data = lines[0].strip().split()
        data = [s.replace("i", "j") for s in data]
        sample_point = int(len(data) / 268)
        time = []
        rssi = []
        agc = []
        csi = []

        for i in range(sample_point):
            time.append(data[i * 268:i * 268 + 3])
            rssi.append(data[i * 268 + 3:i * 268 + 7])
            agc.append(data[i * 268 + 8:i * 268 + 11])
            csi.append([complex(data[x]) for x in range(i * 268 + 12, (i + 1) * 268)])
        real_time = np.zeros(sample_point)
        for i in range(sample_point):
            h, m, s = time[i]
            real_time[i] = int(m) * 60 + float(s)
        for i in range(sample_point):
            timestamp = real_time[i]
            if timestamp - current_timestamp < 2:
                rssi_data.append(rssi[i])
                ag_data.append(agc[i])
                csi_data.append(csi[i])
            else:
                rssi_matrix.append(np.array(rssi_data))
                ag_matrix.append(np.array(ag_data))
                csi_matrix.append(np.array(csi_data))
                rssi_data, ag_data, csi_data = [], [], []
                current_timestamp += 2

        if len(rssi_data):
            rssi_matrix.append(np.array(rssi_data))
            ag_matrix.append(np.array(ag_data))
            csi_matrix.append(np.array(csi_data))
        if csi_label:
            with open(csi_label, 'r', encoding='utf-8') as file:
                labels = [int(k) for k in file.read().split()]
        else:
            labels = [0 for _ in range(len(csi_matrix))]
            # labels = [0 for _ in range(len(fre_res))]
        return rssi_matrix, ag_matrix, csi_matrix, labels