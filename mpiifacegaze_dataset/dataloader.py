import os
import pickle
import numpy as np

import torch
import torch.utils.data

from PIL import Image
from torchvision import transforms

class MPIIFaceGazeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir):
        pickle_file = os.path.join(dataset_dir, 'labels.dict')
        with open(pickle_file, 'rb') as pf:
            d = pickle.load(pf)
            self.images = d['images']
            self.labels = d['labels']
            self.sid = d['subject_id']
        

        self.preprocess = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) 

    def __getitem__(self, index):
        label = self.labels[index][0:2] # Gaze angles only
        img = Image.open(self.images[index])
        img = self.preprocess(img)
        return img, label
    
    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return self.__class__.__name__


def get_loader(dataset_root, batch_size):
    assert os.path.exists(dataset_root)
    
    train_subjects = [os.path.join(dataset_root, '{:02}').format(i) for i in range(15)]
    train_dataset = torch.utils.data.ConcatDataset([
        MPIIFaceGazeDataset(subject) for subject in train_subjects
    ])
    
    assert len(train_dataset) == 45000

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2
    )
    
    return train_loader