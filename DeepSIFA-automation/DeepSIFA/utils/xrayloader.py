import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

    
   
class XrayDataset_train(Dataset):#普通训练用_2分类
    def __init__(self,
                 args):
        df = pd.read_csv(args.train_csv_dir)
        self.npz_list = np.array(df['file_name'])
        self.label_list4 = np.array(df['label'])
        self.data_path = args.data_dir

    def __len__(self):
        return len(self.npz_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.npz_list[index]
        npz_path = os.path.join(self.data_path, name)

        npz_data = np.load(npz_path)
        data = npz_data['data']
        data = data.reshape(1, -1)
        data_tensor = torch.from_numpy(data.astype(np.float32))

        label_cls = self.label_list4[index]
        label_cls_onehot = torch.tensor(np.eye(2)[label_cls])
        return data_tensor, label_cls, label_cls_onehot, name
    
   
class XrayDataset_val(Dataset):#普通验证用_2分类
    def __init__(self,
                 args):
        df = pd.read_csv(args.val_csv_dir)
        self.npz_list = np.array(df['file_name'])
        self.label_list4 = np.array(df['label'])
        self.data_path = args.data_dir

    def __len__(self):
        return len(self.npz_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.npz_list[index]
        npz_path = os.path.join(self.data_path, name)

        npz_data = np.load(npz_path)
        data = npz_data['data']
        data = data.reshape(1, -1)
        data_tensor = torch.from_numpy(data.astype(np.float32))

        label_cls = self.label_list4[index]
        label_cls_onehot = torch.tensor(np.eye(2)[label_cls])
        return data_tensor, label_cls, label_cls_onehot, name

    