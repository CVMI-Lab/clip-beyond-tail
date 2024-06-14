# Data Preparation

- [Data Preparation](#data-preparation)
  - [Creating ImageNet-Captions variants](#creating-imagenet-captions-variants)
    - [Getting started](#getting-started)
    - [Ablating captions](#ablating-captions)
    - [Ablating data scale and concept scale](#ablating-data-scale-and-concept-scale)
    - [Creating trim-tail variants](#creating-trim-tail-variants)
  - [Creating LAIONet variants](#creating-laionet-variants)
    - [Getting started](#getting-started-1)
      - [Option 1: Using our preprocessed metadata](#option-1-using-our-preprocessed-metadata)
      - [Option 2: Recreating LAIONet](#option-2-recreating-laionet)
    - [Ablating intra-class variantion](#ablating-intra-class-variantion)
    - [Ablating data scale](#ablating-data-scale)
    - [Creating LAIONet that matches the frequency distribution of ImageNet-Captions](#creating-laionet-that-matches-the-frequency-distribution-of-imagenet-captions)
  - [Creating YFCC15M-Cls](#creating-yfcc15m-cls)

## Creating ImageNet-Captions variants

We use ImageNet-Captions, which includes 0.45M ImageNet images and paired textual metadata (in the format of title, description, and tags, similar to YFCC-15M) for experiments in this section. The metadata can be downloaded from the [official repo](https://github.com/mlfoundations/imagenet-captions).

### Getting started

Please make sure ImageNet is downloaded to `$DATASET/imagenet` directory. You may consider [this script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) for reference. Then download and unzip [imagenet_captions.zip](https://github.com/mlfoundations/imagenet-captions/blob/main/imagenet_captions.zip) to `$DATASET/imagenet-captions` directory.

### Ablating captions

This ablation requires subsetting ImageNet-Captions with different types of captions. Our script supports including titles, tags, and descriptions. It also supports using a combination of them, or disgard them all and synthesis captions from class names using predefined templates. By default, we suggest setting `append_cname` and `shuffle` to `True` for consistent data scale and random data order. Example scripts are as follows:

```bash
# Full text (title, tags, and description)
python create_incaps_caption_variants.py \
--use_title --use_tags --use_description --append_cname --shuffle

# Template (A {class name})
python create_incaps_caption_variants.py \
--append_cname --mode a+cname --shuffle 

# Template (Synset + OpenAI templates)
python create_incaps_caption_variants.py \
--append_cname --use_synset --mode openai80 --shuffle 
```

### Ablating data scale and concept scale

We support sampling ImageNet-Captions to speficied numer of data (`n`) or number of classes (`k`). Examples are as follows:

```bash
# Create ImageNet-Captions-100
python create_incaps_subsets.py \
--df_path ../datasets/imagenet-captions/incaps_title_tags_description_cname_448896.tsv \
--freq_path ../metadata/freqs/class_frequency_incaps_imagenet_ori.txt \
--use_in100 # or --n 100 if you want to sample classes randomly

# Create ImageNet-Captions (10%)
python create_incaps_subsets.py \
--df_path ../datasets/imagenet-captions/incaps_title_tags_description_cname_448896.tsv \
--freq_path ../metadata/freqs/class_frequency_incaps_imagenet_ori.txt \
--n 44548 # same scale as ImageNet-Caption-100
```

### Creating trim-tail variants

This section creates ImageNet-Captions variants that trim tail classes to few-shot or zero-shot. `lastk` sets the number of tail class to trim, and `ceil` sets the maximum number of tail examples per class. We provide an example script as follows:

```bash
# Create the variant with 1-shot tail of 5 classes
python create_incaps_trim_tail.py \
--df_path ../datasets/imagenet-captions/incaps_title_tags_description_cname_448896.tsv \
--freq_path ../metadata/freqs/class_frequency_incaps_imagenet_ori.txt \
--lastk 5 --ceil 1
```

## Creating LAIONet variants

### Getting started

We first create a full version of LAIONet (with text-def threshold of 0.7), download all images, and create subsets from it for ablations. The metadata can be obtained either by using our preprocessed metadata or created from scratch.
Note that our version adopted a text-def threshold of 0.7 instead of 0.82 as in the original LAIONet paper.

#### Option 1: Using our preprocessed metadata
If no customization is needed, you may use our preprocessed metadata directly from [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xwen_connect_hku_hk/EmcKTKLrlXdDuN7dIZLNRL0B_5KDLAzdStJDHs2YrxW2ag?e=msgTmM).
The files with short names are final metadata that we verified to be downloadable and used in our experiments. For reference, we also provide the full inetemediate medata files (results of Step 4 below, with long names).
Please download the metadata, place them to the [laion400m](laionet-dev/laion400m) directory, and then download and rearrange the images following Steps 5 and 6 below.

#### Option 2: Recreating LAIONet

Please note that considerable disk quota is needed for storing intermediate files.
This includes: 1) finding lemmas of 1000 ILSVRC synsets in LAION-400M (the original parquet files will be downloaded automatically if not found), 2) extracting LAION instances that are matched to at least one lemma, 3) calculating LAION text to synset text similarity, and 4) filtering out the instances with low similarity to their synsets. We provide example shell scripts as follows and please refer to the [official instructions](laionet-dev/readme.md#create-laionet) for more details.

```bash
cd laionet-dev/

# Step 1: Find lemmas of 1000 ILSVRC synsets in LAION-400M
python scripts/createdataset/label_laion_parallel.py

# Step 2: Extract LAION instances that are matched to at least 1 lemma
python scripts/createdataset/subset_laion.py \
--labels_filter "wnid2laionindices(substring_matched_part*).pkl" \
--method substring_matched \
--self_destruct

# Step 3: Calculate LAION text-to-def text similarity
python scripts/calcsimilarity/calc_and_store_clip_text_to_query_similarities.py \
--labels_filter "wnid2laionindices(substring_matched_part*).pkl" \
--method substring_matched \
--query_type name_def \
--query_key wnid \
--gpu_id 0

# Step 4: Filter out the instances with low similarity to their synsets
python scripts/createdataset/sample_laion_from_most_similars.py \
--similarity_col text_to_name_def_wnid_similarity \
--similarity_th 0.7 \
--remove_nsfw \
--method substring_matched_filtered
```

After running the commands above, you will get the filtered LAION-400M metadata in the [laion400m](laionet-dev/laion400m) directory. The labels are stored separately under [laion400m/processed/ilsvrc_labels](laionet-dev/laion400m/processed/ilsvrc_labels), and also merged to the LAIONet dataframe (with column "wnid").
We then download corresponding images using [img2dataset](https://github.com/rom1504/img2dataset), and rearrange the images to a similar stucture like ImageNet. Note that the latter can create ~3M small files, which can be a bottleneck for some file systems. If webdataset format can meet your requirements, you may this step.

```bash
# Step 5: Download the images
cd ../datasets

# set url_list to laionet_thresh07_success3256561.parquet if using our preprocessed metadata
img2dataset --url_list laionet-dev/laion400m/subset_sm_filt_thresh0.7_total4580608_part-00000-to-part00031-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet \ 
--input_format "parquet" \ --url_col "URL" --caption_col "TEXT" --output_format webdataset \
--output_folder laionet-thresh07-data --processes_count 16 --thread_count 64 --image_size 256 \
--resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True \
--save_additional_columns '["NSFW","similarity","wnid","LICENSE"]' \
--enable_wandb True --retries 10

# Step 6: Unzip webdataset, rearrange the images to ImageNet structure, and save the metadata of successfully downloaded images
python arrange_laionet.py \
--dataset_path ../datasets/laionet-thresh07-data/{00000..00458}.tar \
--metadata_path laionet-dev/laion400m/subset_sm_filt_thresh0.7_total4580608_part-00000-to-part00031-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet \
--target_data_path ../datasets/laionet/ \
--target_df_path laionet-dev/laion400m/laionet_thresh0.7.parquet # a postfix of #success images will be appended
```

### Ablating intra-class variantion

This ablation requires subsetting LAIONet with different text-def thresholds. This is motivated by the empirical results that a higher threshold makes images more alike (see [choose_text_query_similarity_threshold.ipynb](laionet-dev/notebooks/choose_text_query_similarity_threshold.ipynb)). We provide an example script to create the subsets with different thresholds.

```bash
python create_laionet_subsets.py \
--df_path laionet-dev/laion400m/laionet_thresh07_success3256561.parquet \
--thresh 0.7 # or 0.75, 0.8, 0.82, etc.
```

### Ablating data scale

This ablation requires subsetting LAIONet with different fractions of images. We provide an example script to create the subsets with different numbers of images.

```bash
python create_laionet_subsets.py \
--df_path laionet-dev/laion400m/laionet_thresh07_success3256561.parquet \
--frac 1 # or 0.5, 0.25, 0.125, 0.0625, 0.03125, etc.
```

### Creating LAIONet that matches the frequency distribution of ImageNet-Captions

This ablation tries to sample a subset from LAIONet that matches the frequency distribution of ImageNet-Captions. Using the one filtered by text-synset similarity solely can hardly achieve this goal due to extreme data imbalance in it. Thus we also recreate ImageNet by searching LAION for most similar captions following [LAIONet instructions](laionet-dev/readme.md#recreate-imagenet-by-searching-laion-for-most-similar-captions) as a supplement. Note that we modified the code to retrieve top-k most similar items, instead to the top-1 only.
Also, please make sure you have downloaded and preprocessed ImageNet-Captions following [this guideline](laionet-dev/imagenet-captions/readme.md).

```bash
cd laionet-dev/

# Step 1: Find the most similar texts to ImageNet-Captions
python scripts/searchtext/find_most_similar_texts_to_texts.py \
--dataframe_path imagenet-captions/imagenet_captions.parquet \
--topk 50 \
--gpu_id 0

# Step 2: Create inverse index from wnid to LAION indices
python scripts/preprocess/find_wnid_to_laion_indices_local.py

# Step 3: Sample the dataset
python scripts/createdataset/subset_laion.py \
--labels_filter "wnid2laionindices(subset_ic_most_sim_txttxt)_top50.pkl" \
--method imagenet_captions_most_similar_text_to_texts \
--self_destruct

# Step 4: Download the images
cd ../datasets

# set url_list to laionet_t2t_success309113.parquet if using our preprocessed metadata
img2dataset --url_list laionet-dev/laion400m/subset_ic_most_sim_txttxt_top50_total424471_part-00000-to-part00031-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet \
--input_format "parquet" --url_col "URL" --caption_col "TEXT" --output_format webdataset \
--output_folder laionet-t2t-data --processes_count 16 --thread_count 64 --image_size 256 \
--resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True \
--save_additional_columns '["NSFW","similarity","wnid","LICENSE"]' \
--enable_wandb True --retries 10

# Step 5: Unzip webdataset, rearrange the images to ImageNet structure, and save the metadata of successfully downloaded images
python arrange_laionet.py \
--dataset_path ../datasets/laionet-t2t-data/{00000..0042}.tar \
--metadata_path laionet-dev/laion400m/subset_ic_most_sim_txttxt_top50_total424471_part-00000-to-part00031-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet
--target_data_path ../datasets/laionet/ \
--target_df_path laionet-dev/laion400m/laionet_t2t_top50.parquet # a postfix of #success images will be appended

# Step 6: Create matching IN-Caps variant from merged data sources
python create_laionet_match_freq.py \
--df_path laionet-dev/laion400m/laionet_thresh07_success3256561.parquet \
--t2tdf_path laionet-dev/laion400m/laionet_t2t_success309113.parquet \
--capdf_path laionet-dev/imagenet-captions/imagenet_captions.parquet
```

## Creating YFCC15M-Cls

YFCC-15M-Cls is sampled from YFCC-15M by matching captions to ImageNet synsets, and no other filtering is applied. One difference from [Fang et al.](https://arxiv.org/pdf/2205.01397) is that our implementation is not case-sensitive.
To run our script, please first make sure YFCC-15M is downloaded. In our case, there is a jsonl file containing the image paths and paired captions as input, and it can be easily adapted to other formats. We provide an example script as follows:

```bash
# You may modify the paths in the script to create CC12M-Cls, etc.
python create_yfcc15m_cls.py
```