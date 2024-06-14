import os
import argparse
import logging
import pandas as pd
import webdataset as wds

import multiprocessing
from tqdm import tqdm

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def process_item(item, target_data_path):
    key, image, text = item
    key = int(key)
    if key in passed_keys:
        return

    wnid = indice2wnids[key]
    laion_indice = indice2laion_indices[key]
    imgpath = target_data_path + f"{wnid}/{laion_indice}.jpg"
    if os.path.exists(imgpath):
        return

    os.makedirs(target_data_path + f"{wnid}", exist_ok=True)
    image.save(imgpath)
    # with open(target_data_path + f"{wnid}/{laion_indice}.txt", "w") as f:
    #     f.write(text)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../datasets/laionet-thresh07-data/{00000..00458}.tar")
    parser.add_argument("--metadata_path", type=str, default="laionet-dev/laion400m/subset_sm_filt_thresh0.7_total4580608_part-00000-to-part00031-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet")
    parser.add_argument('--target_data_path', type=str, default='../datasets/laionet/')
    parser.add_argument("--target_df_path", type=str, default="laionet-dev/laion400m/laionet_thresh07.parquet")
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    dataset = wds.WebDataset(args.dataset_path).decode("pilrgb", handler=log_and_continue).rename(key="__key__", image="jpg;png;jpeg;webp", text="txt").to_tuple("key", "image", "text")
    df = pd.read_parquet(args.metadata_path)
    indice2laion_indices = list(df.index)
    indice2wnids = list(df['wnid'])

    passed_keys = set()
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for i, item in tqdm(enumerate(dataset)):
            if i >= len(df):
                break
            pool.apply_async(process_item, (item, args.target_data_path))
            passed_keys.add(int(item[0]))
    
    print(len(passed_keys))
    passed_keys = list(sorted(passed_keys))
    success_df = df.iloc[passed_keys]
    output_path = args.target_df_path.replace(".parquet", "_success{}.parquet".format(len(passed_keys)))
    success_df.to_parquet(output_path)