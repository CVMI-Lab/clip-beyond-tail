import os
import json
import csv
import argparse
import numpy as np
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_path', type=str, default='../datasets/imagenet-captions/incaps_title_tags_description_cname_448896.tsv')
    parser.add_argument('--freq_path', type=str, default='../metadata/freqs/class_frequency_incaps_imagenet_ori.txt')
    parser.add_argument('--k', type=int, default=None, help='number of classes to keep')
    parser.add_argument('--n', type=int, default=None, help='number of samples to keep')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--use_in100', action='store_true', default=False, help='use imagenet-100 classes')
    args = parser.parse_args()

    assert args.k is not None or args.use_in100 or args.n is not None

    classnames = list(json.load(open('../metadata/descriptors/descriptors_imagenet.json', 'r')).keys())
    df = pd.read_csv(args.df_path, sep='\t', quoting=csv.QUOTE_NONE)
    save_path = args.df_path
    freq_save_path = args.freq_path

    if args.use_in100:
        cls_idxs = np.loadtxt('../metadata/imagenet100_idxs.txt', dtype=int).tolist()
    elif args.k is not None:
        assert args.k > 1 and args.k <= 1000
        np.random.seed(args.seed)
        cls_idxs = np.random.choice(1000, args.k, replace=False).tolist()
    else:
        cls_idxs = None
    
    if cls_idxs is not None:
        cls_idxs = sorted(cls_idxs)
        df = df[df['label'].isin(cls_idxs)]
        df['label_ori'] = df['label'] # keep original labels
        label_map = {cls_idx: i for i, cls_idx in enumerate(cls_idxs)}
        df['label'] = df['label_ori'].map(label_map) # remap labels
        classnames = classnames[cls_idxs]
        save_path = save_path.replace('incaps', 'in{}caps'.format(len(cls_idxs)))
        freq_save_path = freq_save_path.replace('incaps', 'in{}caps'.format(len(cls_idxs)))

    if args.n is not None:
        assert args.n > 1 and args.n <= len(df)
        df = df.sort_values(by='filepath')
        df = df.sample(n=args.n, random_state=args.seed)
        save_path = save_path[:save_path.rfind('_')] + '_{}.tsv'.format(args.n)
        freq_save_path = freq_save_path.replace('incaps', 'incaps_n{}'.format(args.n))

    # iter over df
    id2counts = {id:0 for id in range(len(classnames))}
    for id in df['label']:
        id2counts[id] += 1

    print(df)

    df.to_csv(save_path, sep='\t', index=False)
    freq_df = pd.DataFrame({'classname': classnames, 'counts': list(id2counts.values())})
    freq_df.to_csv(freq_save_path, sep='\t', index=False, header=False)