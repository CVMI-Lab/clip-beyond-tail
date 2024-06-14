import json
import argparse
import pandas as pd
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lastk', type=int, default=5)
    parser.add_argument('--ceil', type=int, default=5)
    parser.add_argument('--df_path', type=str, default='../datasets/imagenet-captions/incaps_title_tags_description_cname_448896.tsv')
    parser.add_argument('--freq_path', type=str, default='../metadata/freqs/class_frequency_incaps_imagenet_ori.txt')
    args = parser.parse_args()

    classnames = list(json.load(open('../metadata/descriptors/descriptors_imagenet.json', 'r')).keys())

    df = pd.read_csv(args.df_path, sep='\t')

    # iter over df
    counts = [0] * len(classnames)
    for lab in df['label']:
        counts[lab] += 1

    tail_classes = np.argsort(counts)[:args.lastk]
    for cls in tail_classes:
        subdf = df[df['label'] == cls]
        if len(subdf) > args.ceil:
            drop_indices = np.random.choice(subdf.index, len(subdf) - args.ceil, replace=False).tolist()
            df = df.drop(drop_indices)
            counts[cls] = args.ceil

    df.to_csv(args.df_path[:args.df_path.rfind('_')] + '_tail{}ceil{}.tsv'.format(args.lastk, args.ceil), sep='\t', index=False)
    
    if args.freq_path != 'None':
        freq_df = pd.read_csv(args.freq_path, sep='\t', header=None, names=['classname', 'counts'])
        freq_df['counts'] = counts
        freq_df.to_csv(args.freq_path.replace('incaps', 'incaps_tail{}ceil{}'.format(args.lastk, args.ceil)), sep='\t', header=False, index=False)
