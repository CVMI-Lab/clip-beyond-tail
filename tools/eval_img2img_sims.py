# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torchvision import transforms as transforms
from open_clip import create_model_and_transforms

from sup_exps.utils import BICUBIC, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from sup_exps.model import ClsWrapper
from sup_exps.dataset import CsvDatasetTrim

@torch.no_grad()
def get_sims(model, dataloader, args):
    all_features, all_targets = [], []
    all_sims = {}
    model.eval()
    for images, target in tqdm(dataloader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        if args.model[0].isupper():
            output = model(image=images)
            image_features = output['image_features'] if isinstance(output, dict) else output[0]
        else:
            image_features = model.encode_image(images)
        all_features.append(image_features)
        all_targets.append(target)
    all_features, all_targets = torch.cat(all_features), torch.cat(all_targets)
    # put feathres to the corresponding class
    for i in range(args.num_classes):
        features = all_features[all_targets == i]
        if features.shape[0] > 0:
            feature = nn.functional.normalize(features, dim=1, p=2)
            sims = torch.mm(feature, feature.T).view(-1)
            all_sims[i] = sims.cpu().numpy()
        else:
            all_sims[i] = np.array([])
    return all_sims

def get_parser():
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--model', default='RN50', type=str, help='Architecture')
    parser.add_argument("--pretrained", type=str, default='openai', help="Use a pretrained CLIP model weights with the specified tag or file path.")
    parser.add_argument("--upper", type=int, default=100)
    parser.add_argument('--data_path', default='./datasets/imagenet', type=str)

    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')

    parser.add_argument('--log_dir', default="./results", help='Path to save logs and checkpoints.')
    parser.add_argument('--num_classes', default=1000, type=int, help='Number of labels for linear classifier.')
    args = parser.parse_args()
    return args

def main(args):
    # ============ building network ... ============
    if args.model[0].isupper(): # is from openclip
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            args.model,
            args.pretrained,
            device='cuda',
            output_dict=True,
        )
    else:
        model = ClsWrapper(args.model, args.num_classes)
        preprocess_val = transforms.Compose([
            transforms.Resize(256, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
        ])
    model.cuda()
    model.eval()


    # ============ preparing data ... ============
    dataset = CsvDatasetTrim(args.data_path, upper=args.upper, transform=preprocess_val)
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1024,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_sims = get_sims(model, val_loader, args)
    dname = os.path.basename(args.data_path)[:-4]  # .split('.')[0]
    torch.save(all_sims, os.path.join(args.log_dir, f'{dname}_sims.pth'))

if __name__ == '__main__':
    args = get_parser()
    main(args)