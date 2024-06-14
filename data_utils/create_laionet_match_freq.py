import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

pd.options.mode.chained_assignment = None  # default='warn'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_path', type=str, default='laionet-dev/laion400m/laionet_thresh07_success3256561.parquet')
    parser.add_argument('--t2tdf_path', type=str, default='laionet-dev/laion400m/laionet_t2t_success309113.parquet')
    parser.add_argument('--cap_path', type=str, default='laionet-dev/imagenet-captions/imagenet_captions.parquet')
    args = parser.parse_args()

    wnids = sorted(json.load(open("../metadata/descriptors/descriptors_imagenet_synset.json", "r")).keys())
    classnames = list(json.load(open('../metadata/descriptors/descriptors_imagenet.json', 'r')).keys())
    wnid2classname = dict(zip(wnids, classnames))
    wnid2label = {wnid: i for i, wnid in enumerate(wnids)}

    df = pd.read_parquet(args.df_path)
    t2t_df = pd.read_parquet(args.t2tdf_path)
    cap_df = pd.read_parquet(args.cap_path)

    # iter over df
    wnid2counts = {wnid:0 for wnid in wnids}
    for index in cap_df.index:
        wnid = index.split('_')[0]
        wnid2counts[wnid] += 1
    
    dfs, cur_cnts = {}, {wnid:0 for wnid in wnids}
    
    for wnid in tqdm(wnids):
        subdf = df[df['wnid'] == wnid]
        if len(subdf) > wnid2counts[wnid]:
            sample_indices = np.random.choice(subdf.index, wnid2counts[wnid], replace=False).tolist()
            subdf = subdf.loc[sample_indices]
        
        subdf['from'] = 'laionet'
        cur_cnts[wnid] = len(subdf)
        dfs[wnid] = subdf
        
        if cur_cnts[wnid] == wnid2counts[wnid]:
            continue
        
        subdf_t2t = t2t_df[t2t_df['wnid'] == wnid]
        tocollect = wnid2counts[wnid] - cur_cnts[wnid]
        if len(subdf_t2t) > tocollect:
            sample_indices = np.random.choice(subdf_t2t.index, tocollect, replace=False).tolist()
            subdf_t2t = subdf_t2t.loc[sample_indices]
        else:
            print('wnid {} has {} samples, less than {}'.format(wnid, len(subdf_t2t)+cur_cnts[wnid], wnid2counts[wnid]))
        
        subdf_t2t['from'] = 'laionet-t2t'
        cur_cnts[wnid] += len(subdf_t2t)
        dfs[wnid] = pd.concat([dfs[wnid], subdf_t2t])
    
    new_df = pd.concat(list(dfs.values()))
    new_df = new_df.sort_index()

    labels = [wnid2label[wnid] for wnid in new_df['wnid']]
    filepaths = ["../datasets/{}/{}/{}.jpg".format(frm, wnid, index) for frm, wnid, index in zip(new_df['from'], new_df['wnid'], new_df.index)]
    new_df = new_df[['TEXT', 'wnid']]
    new_df = new_df.assign(label=labels, filepath=filepaths)
    new_df = new_df.rename(columns={'TEXT': 'caption'})
    new_df = new_df[['filepath', 'label', 'caption']]
    new_df['caption'] = new_df['caption'].apply(lambda x: x.replace('\r', ' ').replace('\n', ' ').replace('\t', ' '))

    print(new_df)

    new_df.to_csv('../datasets/imagenet-captions/laionet_match_incaps_{}.tsv'.format(len(new_df)), sep='\t', index=False)

    freq_df = pd.DataFrame({'classname': classnames, 'counts': list(cur_cnts.values())})
    freq_df.to_csv('../metadata/freqs/class_frequency_laionet_match_incaps_ori.txt', sep='\t', index=False, header=False)
    