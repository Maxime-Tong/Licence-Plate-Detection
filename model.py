import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PNet(nn.Module):
    def __init__(self, N_classes) -> None:
        super(PNet, self).__init__()
        self.N_classes = N_classes
        hidden_size = 5 * 5 * 16
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten()
        )
        # [1, 20, 20] -> [10, 10, 6] -> [5, 5, 16] -> [5, 5, 32]
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, N_classes)
        )

    def forward(self, data):
        # bs, ch, H, W = data.shape
        hidden = self.conv(data)
        out = self.linear(hidden)
        return out
    
        

class Licence(Dataset):
    def __init__(self, img_path, labels_map, img_aug):
        super(Licence, self).__init__()
        self.root = img_path
        self.labels_map = labels_map
        self.labels2id = {v: i for i, v in enumerate(labels_map.keys())}
        self.id2labels = {v: k for k, v in self.labels2id.items()}
        print(self.id2labels)
        
        data = []
        labels = []
        N_classes = len(labels_map)
        for label in self.labels_map:
            folder = os.path.join(self.root, label)
            for file in os.listdir(folder):
                img_path = os.path.join(folder, file)
                img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (20, 20))
                img = np.expand_dims(img, axis=0)
                
                # Nd = 0.2
                # Sd = 1 - Nd
                # mask = np.random.choice((0, 1, 2), size=(1, 20, 20), p=[Nd/2.0, Nd/2.0, Sd])
                # img[mask == 0] = 0
                # img[mask == 1] = 255
                
                data.append(img)
                labels.append(F.one_hot(torch.tensor(self.labels2id[label]), num_classes=N_classes))
                
        self.data = torch.as_tensor(np.stack(data, axis=0) / 255).float()
        self.labels = torch.stack(labels, dim=0).float()
        
        if img_aug:
            trans = transforms.RandomAffine(degrees=10, scale=(0.8, 1.2), fill=0)
            self.data = trans(self.data)
        # print(self.data.shape)
        # print(self.labels.shape)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, item):
        img = self.data[item]
        label = self.labels[item]
        return img, label
