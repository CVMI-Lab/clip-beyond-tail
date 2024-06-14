# Running CLIP Experiments

- [Running CLIP Experiments](#running-clip-experiments)
  - [Getting started](#getting-started)
    - [Data preparation](#data-preparation)
    - [Environment setup](#environment-setup)
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

Our CLIP experiments are conducted using a customized version of [open_clip](./open_clip/), which is included as submodule. If it is not cloned together with this repository, you may run the `git submodule update --init --recursive` in the root directory to fetch it. Then, run the following commands to install it from source:

```bash
cd open_clip
make install
make install-training
```

## Running

### Training

We privide example scripts to replicate our experiments in the [scripts](./scripts/) folder. This includes investigations on [caption descriptiveness](./scripts/ablate_caption) (Sec. 3.2), [data distribution](./scripts/ablate_distribution/), [diversity](./scripts/ablate_diversity/), [imbalance level](./scripts/ablate_imbalance/) (Sec. 3.4), [data scale](./scripts/ablate_scale/) (Sec. 3.5) and [open-world concepts](./scripts/ablate_concept_set/) (Sec. 3.6). It also supports explorations on [few-shot and open-world recognition](./scripts/trim_tail/) (Sec. 4.1). You may run them directly or modify them to suit your needs. Checkpoints and intermediate evaluation results are saved to the [logs](./logs/) folder by default.

### Evaluation

The metrics are already computed and saved during training. We also provide example scripts for re-evaluating trained checkpoints, e.g., for additional evaluation datasets or prompts. The following is an example:

```bash
export TEMPLATE_TYPE=openai # open_clip default option
bash scripts/zero_shot/run_openzlip_zs.sh $PATH_TO_RUN_FOLDER $TEMPLATE_TYPE
```

The results will be saved to the same directory as the checkpoint file.

Besides, we also support evaluating all pre-trained CLIP models provided by [open_clip](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md). Simply run `bash scripts/zero_shot/run_all.sh` or `python run_pretrained_openclip.py` to evaluate them, and check [run_pretrained_openclip.py](./run_pretrained_openclip.py) for details. The results will be saved to [logs_pretrained](./logs_pretrained/) by default.