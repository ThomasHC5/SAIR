import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
import numpy as np

def load_data(eeg_path, img_path, nSub, seed, batch_size, batch_size_test):
    train_dataset = EEGDataset_train(eeg_path, img_path, nSub)
    val_dataset = EEGDataset_val(eeg_path, img_path, nSub)
    test_dataset = EEGDataset_test(eeg_path, nSub)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False)
    
    test_edge, test_depth, test_scene = get_test_img_features(img_path)

    return train_loader, val_loader, test_loader, test_edge, test_depth, test_scene

def get_test_img_features(img_path):
    test_edge = np.load(img_path + 'edge_test.npy', allow_pickle=True)
    test_depth = np.load(img_path + 'depth_test.npy', allow_pickle=True)
    test_scene = np.load(img_path + 'blur_test.npy', allow_pickle=True)

    test_edge = np.squeeze(test_edge)
    test_depth = np.squeeze(test_depth)
    test_scene = np.squeeze(test_scene)

    test_edge = torch.from_numpy(test_edge)
    test_depth = torch.from_numpy(test_depth)
    test_scene = torch.from_numpy(test_scene)

    return test_edge, test_depth, test_scene

class EEGDataset_train(TensorDataset):
    def __init__(self, eeg_path, img_path, nSub):
        self.eeg_path = eeg_path
        self.img_path = img_path
        self.nSub = nSub
        self.data = self.load_eeg()
        self.img_edge, self.img_depth, self.img_scene = self.load_img()

    def load_eeg(self):
        data_list = []
        for sub in range(1, 11):
            if sub == self.nSub:
                continue
            data_file = self.eeg_path + '/sub-' + format(sub, '02') + '/preprocessed_eeg_training.npy'
            data = np.load(data_file, allow_pickle=True)
            data = data['preprocessed_eeg_data']
            data = np.mean(data, axis=1)
            data = np.expand_dims(data, axis=1)
            data_list.append(data)
        data = np.concatenate(data_list, axis=0)
        data = torch.from_numpy(data)
        return data
    
    def load_img(self):
        img_edge = np.load(self.img_path + 'edge_training.npy', allow_pickle=True)
        img_depth = np.load(self.img_path + 'depth_training.npy', allow_pickle=True)
        img_scene = np.load(self.img_path + 'blur_training.npy', allow_pickle=True)

        img_edge = np.squeeze(img_edge)
        img_depth = np.squeeze(img_depth)
        img_scene = np.squeeze(img_scene)

        img_edge = torch.from_numpy(img_edge)
        img_depth = torch.from_numpy(img_depth)
        img_scene = torch.from_numpy(img_scene)

        return img_edge, img_depth, img_scene

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_idx = idx % 16540
        sample = self.data[idx]
        label_edge = self.img_edge[img_idx]
        label_depth = self.img_depth[img_idx]
        label_scene = self.img_scene[img_idx]
        return sample, label_edge, label_depth, label_scene

class EEGDataset_val(TensorDataset):
    def __init__(self, eeg_path, img_path, nSub):
        self.eeg_path = eeg_path
        self.img_path = img_path
        self.nSub = nSub
        self.data = self.load_eeg()
        self.img_edge, self.img_depth, self.img_scene = self.load_img()

    def load_eeg(self):
        data_list = []
        for sub in range(1, 11):
            if sub == self.nSub:
                continue
            data_file = self.eeg_path + '/sub-' + format(sub, '02') + '/preprocessed_eeg_test.npy'
            data = np.load(data_file, allow_pickle=True)
            data = data['preprocessed_eeg_data']
            data = np.mean(data, axis=1)
            data = np.expand_dims(data, axis=1)
            data_list.append(data)
        data = np.concatenate(data_list, axis=0)
        data = torch.from_numpy(data)
        return data
    
    def load_img(self):
        img_edge = np.load(self.img_path + 'edge_test.npy', allow_pickle=True)
        img_depth = np.load(self.img_path + 'depth_test.npy', allow_pickle=True)
        img_scene = np.load(self.img_path + 'blur_test.npy', allow_pickle=True)

        img_edge = np.squeeze(img_edge)
        img_depth = np.squeeze(img_depth)
        img_scene = np.squeeze(img_scene)

        img_edge = torch.from_numpy(img_edge)
        img_depth = torch.from_numpy(img_depth)
        img_scene = torch.from_numpy(img_scene)

        return img_edge, img_depth, img_scene

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_idx = idx % 200
        sample = self.data[idx]
        label_edge = self.img_edge[img_idx]
        label_depth = self.img_depth[img_idx]
        label_scene = self.img_scene[img_idx]
        return sample, label_edge, label_depth, label_scene

class EEGDataset_test(TensorDataset):
    def __init__(self, eeg_path, nSub):
        self.eeg_path = eeg_path
        self.nSub = nSub
        self.data = self.load_eeg()
        self.labels = torch.from_numpy(np.arange(200))

    def load_eeg(self):
        data_file = self.eeg_path + '/sub-' + format(self.nSub, '02') + '/preprocessed_eeg_test.npy'
        data = np.load(data_file, allow_pickle=True)
        data = data['preprocessed_eeg_data']
        data = np.mean(data, axis=1)
        data = np.expand_dims(data, axis=1)
        data = torch.from_numpy(data)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label