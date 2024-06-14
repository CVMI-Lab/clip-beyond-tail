import os
import json
import glob
import torch
import pandas as pd

from PIL import Image

# imagenet-caption dataset for classification task
class ImageNetCaptionCls(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(ImageNetCaptionCls, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # set root to imagenet-1k, and wnids are the 1000 classes
        paths = list(glob.glob(os.path.join(self.root, self.split, '*')))
        self.wnids = sorted([path.split('/')[-1] for path in paths if path.split('/')[-1].startswith('n')])
        self.wnid_to_label = {wnid: i for i, wnid in enumerate(self.wnids)}

        metadata = json.load(open(os.path.join(self.root, 'imagenet_captions.json')))
        self.data = [os.path.join(root, split, d['wnid'], d['filename']) for d in metadata if d['wnid'] in self.wnids]
        self.targets = [self.wnid_to_label[d['wnid']] for d in metadata if d['wnid'] in self.wnids]

    def __getitem__(self, index):
        img = Image.open(self.data[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.targets[index]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.data)


class CsvDataset(torch.utils.data.Dataset):
    def __init__(self, input_filename, transform=None, target_transform=None, img_key='filepath', target_key='label', sep="\t"):
        print(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.targets = df[target_key].tolist()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        img = Image.open(img_path).convert('RGB')
        label = int(self.targets[idx])
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label


class CsvDatasetTrim(torch.utils.data.Dataset):
    def __init__(self, input_filename, upper=100, transform=None, target_transform=None, img_key='filepath', target_key='label', sep="\t"):
        print(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
        df = df.groupby(target_key).head(upper)
        
        self.images = df[img_key].tolist()
        self.targets = df[target_key].tolist()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        img = Image.open(img_path).convert('RGB')
        label = int(self.targets[idx])
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label