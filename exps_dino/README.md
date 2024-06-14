# Running DINO Experiments

This folder includes code for training and evaluating DINO variants based on customizations of [dino](https://github.com/facebookresearch/dino) and [ssl-transfer](https://github.com/linusericsson/ssl-transfer).

- [Running DINO Experiments](#running-dino-experiments)
  - [Getting started](#getting-started)
    - [Data preparation](#data-preparation)
    - [Environment setup](#environment-setup)
  - [Running](#running)
    - [Training](#training)
    - [Evaluation](#evaluation)

## Getting started

### Data preparation

We support training DINO on folder-based image datasets (e.g., ImageNet and rearranged LAIONet) and csv-based datasets (e.g., ImageNet-Captions and LAIONet subsets). The example scripts use the former by default.

For transfer learning evaluation, we support 12 downstream datasets, including CIFAR10/100, Aircraft, Birdsnap, Caltech101, Cars, DTD, Flowers, Food, Pets, SUN397, and VOC. Please download the datasets following the instructions in [ssl-transfer](./ssl-transfer/readme.md#datasets), and the resulting file structure should follow details in [config.py](./ssl-transfer/config.py).

### Environment setup

Existing environment used for SL and CLIP training should be sufficient. Please refer to the [README](./ssl-transfer/readme.md#requirements) for reference.

## Running

### Training

We privide example scripts to replicate our experiments in the [scripts](./scripts/) folder. This includes training DINO on ImageNet and LAIONet, with different numbers of prototypes (fixed), or different numbers of prototypes to sample from the 65536 ones when computing the loss. We train with 8 GPUs by default. Checkpoints are saved to the [output](./output/) folder by default.

### Evaluation

After training is done, first convert the checkpoint to desiered format by running the following command:

```bash
python convert_dino_ckpt.py --model $PATH_TO_CHECKPOINT
```
By default, the converted checkpoint named `dino.pth` will be saved to the same directory as the original checkpoint.
Then, run the following command to evaluate the checkpoint on downstream datasets with linear probing:

```bash
python scripts/run_linear_transfer.sh $PATH_TO_CHECKPOINT $DATASET_NAME # e.g., cifar10
```

The results will be saved to `linear.csv` in the same directory as the checkpoint file.

Note that the scripts also supports evaluating SL and CLIP checkpoints, and few-shot/fully fine-tune evaluations. Feel free to explore the code.