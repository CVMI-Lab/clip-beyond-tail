import argparse
import pandas as pd
import pickle as pk

from tqdm import tqdm


if __name__ == "__main__":
    df_path = "laionet-dev/laion400m/subset_ic_most_sim_txttxt_{}_part-00000-to-part00031-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet".format("nothresh")
    df = pd.read_parquet(df_path)
    
    index2laionindices = pk.load(open("laionet-dev/laion400m/processed/ilsvrc_labels/icimagename2laionindices_top50.pkl", "rb"))
    index2sims = pk.load(open("laionet-dev/laion400m/processed/ilsvrc_labels/icimagename2sims_top50.pkl", "rb"))
    index2wnid = pk.load(open("laionet-dev/imagenet-captions/processed/labels/icimagename2wnid.pkl", "rb"))
    
    laionindex2sims = {}
    for index, laionindices, sims in tqdm(zip(index2laionindices.keys(), index2laionindices.values(), index2sims.values())):
        wnid = index2wnid[index]
        for laionindex, sim in zip(laionindices, sims):
            if laionindex not in laionindex2sims:
                laionindex2sims[laionindex] = {}
            laionindex2sims[laionindex][wnid] = sim
    
    # get most similar wnid for each laionindex
    for laionindex in tqdm(df.index):
        df.loc[laionindex, "wnid"] = max(laionindex2sims[laionindex], key=laionindex2sims[laionindex].get)
    
    df.to_parquet(df_path)
