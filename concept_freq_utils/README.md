# Concept Frequency Estimation

This folder contains the code for estimating the frequency of concepts in a image-text dataset. The code is written at a very early stage of this project, thus is not well-organized and not optimized. We provide the results in [metadata/freqs](../metadata/freqs) folder for CC-12M, YFCC-15M, LAION-400M, LAION-2B, and MetaCLIP. For future usage, we recommend just take the code for reference, and consider latest works in this direction, e.g., [MetaCLIP](https://github.com/facebookresearch/MetaCLIP), [NeglectedTailsVLM](https://github.com/shubhamprshr27/NeglectedTailsVLM), and [frequency_determines_performance](https://github.com/bethgelab/frequency_determines_performance).

## Requirements

`nltk` is required for word tokenization and lemmatization, `pandas` is needed for loading image-text dataset metadata, and `tqdm` is used for progress bar. You can install them via pip. Besides, processing large-scale datasets requires considerable CPU cores and memory, we recommend using a high-performance server. Also, saving the tokenized intermediate results can take up a lot of disk space, be aware of that.

## Usage

Download the metadata of image-text datasets containing the captions, e.g., [CC-12M](https://storage.googleapis.com/conceptual_12m/cc12m.tsv), [YFCC-15M](https://gitlab.com/jfolz/yfcc100m/-/issues/2), [LAION-400M](https://laion.ai/laion-400-open-dataset/), [LAION-2B](https://huggingface.co/datasets/laion/laion2B-en), [MetaCLIP-400M](https://github.com/facebookresearch/MetaCLIP/blob/main/metaclip/datacard_400m.json), and [MetaCLIP-2.5B](https://github.com/facebookresearch/MetaCLIP/blob/main/metaclip/datacard_fullcc2.5b.json).

The following command calculates the frequency of ImageNet classes in the LAION-400M dataset. You can replace the `--url_path` with the path to the metadata file of other datasets, and specify the `--dataset` to indicate which dataset the concepts are from. The `--input_format` is used to specify the format of the metadata file, which can be `tsv`, `parquet`, or `json`.

```bash
python calc_class_frequency.py \
--url_path ../datasets/laion/laion400m-meta \
--input_format parquet \
--caption_col TEXT \
--dataset imagenet # which dataset the concepts of interest are from
```

For MetaCLIP, you should run the following command because the authors has provided a file of concept frequency.

```bash
python calc_class_frequency_metaclip.py \
--json_path ../datasets/MetaCLIP/metaclip/datacard_400m.json \
--dataset imagenet
```

We also provide the code for calculating the frequency of words instead of class names. You may check [calc_word_frequency.py](calc_word_frequency.py) for more details and the usage is similar to the above.