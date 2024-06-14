# Supervised Learning Experiments

- [Supervised Learning Experiments](#supervised-learning-experiments)
  - [Getting started](#getting-started)
    - [Data preparation](#data-preparation)
    - [Environment setup](#environment-setup)
    - [Pre-trained heads](#pre-trained-heads)
  - [Running](#running)
    - [Training](#training)
    - [Evaluation](#evaluation)

## Getting started

### Data preparation

Please make sure ImageNet is downloaded to `$DATASET/imagenet` directory. Then, you may create training dataset variants of ImageNet-Captions, LAIONet, YFCC-15M, and CC-12M by following the instructions in the [data_preparation](../data_preparation/README.md) folder. TSV files containing image paths will be stored under `$DATASET/imagenet-captions`, and corresponding class frequencies will be stored under [freqs](../metadata/freqs/) folder.

Evaluation is done on the ImageNet validation set, which is expected to be stored under `$DATASET/imagenet/val`. Optionally, we also support evaluating on [ImageNetV2](https://github.com/modestyachts/ImageNetV2) and [ImageNet-100](https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt). Example commands to download these datasets are provided below.

```bash
export DATASET=../datasets

# Download ImageNetV2
mkdir $DATASET/imagenetv2 && cd $DATASET/imagenetv2
wget https://huggingface.co/datasets/vaishaal/ImageNetV2/blob/main/imagenetv2-matched-frequency.tar.gz
tar -xvf imagenetv2-matched-frequency.tar.gz
rm imagenetv2-matched-frequency.tar.gz

# Download ImageNet-100
mkdir $DATASET/imagenet100 && cd $DATASET/imagenet100
git clone https://github.com/danielchyeh/ImageNet-100-Pytorch.git && cd ImageNet-100-Pytorch
python generate_IN100.py --source_folder $DATASET/imagenet --target_folder $DATASET/imagenet100
rm -r $DATASET/imagenet100/ImageNet-100-Pytorch
```

### Environment setup

Nothing to take special care here. Basically just make sure PyTorch (>=2.0, with CUDA) is installed and there are at least 4 GPUs on your device.

### Pre-trained heads

We have provided pre-extracted class embeddings for 1K ImageNet classes with different prompts and text encoders, check [heads](../metadata/heads/) folder for details. You may also extract your own class embeddings using [dump_clip_txt_features.py](../tools/dump_clip_txt_features.py). Depending the text encoder you use, you may need to install corresponding libraries, e.g., [clip](https://github.com/openai/clip), [open_clip](https://github.com/mlfoundations/open_clip), and [transformers](https://github.com/huggingface/transformers).

## Running

### Training

We privide example scripts to replicate our experiments in the [scripts](./scripts/) folder. This includes investigations on [vocabulary size](./scripts/ablate_voc/) (Sec. 3.3), [data distribution](./scripts/ablate_distribution/) (Sec. 3.4 & 3.5), and [open-world concepts](./scripts/ablate_concept_set/) (Sec. 3.6). It also supports explorations on [few-shot and open-world recognition](./scripts/trim_tail/) (Sec. 4.1). You may run them directly or modify them to suit your needs. Checkpoints and intermediate evaluation results are saved to the [output](./logs/) folder by default.

### Evaluation

The metrics are already computed and saved during training. If you want to re-evaluate a trained model, you may run the following command:

```bash
bash scripts/eval.sh $PATH_TO_CHECKPOINT
```

The results will be saved to the same directory as the checkpoint file.