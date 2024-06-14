import json
import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_path', type=str, default='laionet-dev/laion400m/laionet_thresh07_success3256561.parquet')
    parser.add_argument('--thresh', type=float, default=0.7, help='text-to-definition similarity threshold')
    parser.add_argument('--frac', type=float, default=1, help='fraction of samples to keep')
    parser.add_argument('--n', type=int, default=None, help='number of samples to keep')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    wnids = sorted(json.load(open("../metadata/descriptors/descriptors_imagenet_synset.json", "r")).keys())
    classnames = list(json.load(open('../metadata/descriptors/descriptors_imagenet.json', 'r')).keys())
    wnid2classname = dict(zip(wnids, classnames))
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    df = pd.read_parquet(args.df_path)
    one_per_cls = df.groupby('wnid').head(1)
    df = df[df['text_to_name_def_wnid_similarity'] >= args.thresh]

    if args.n is not None:
        df = df.sample(n=args.n, random_state=args.seed)
    else:
        df = df.sample(frac=args.frac, random_state=args.seed)
    
    # ensure number of classes do not decrease
    wnids_to_take = set(wnids) - set(df['wnid'])
    df = pd.concat([df, one_per_cls[one_per_cls['wnid'].isin(wnids_to_take)]])
    if args.n is not None and len(df) > args.n:
        # randomly drop samples from top classes to ensure total n samples
        num_sample_to_drop = len(df) - args.n
        # wnids that have most samples
        top_wnids = df['wnid'].value_counts().index.tolist()[:10]
        # randomly drop num_sample_to_drop samples from top classes
        idxs_to_drop = df[df['wnid'].isin(top_wnids)].sample(n=num_sample_to_drop, random_state=args.seed).index
        df = df.drop(idxs_to_drop)

    df = df.sort_index()

    # iter over df
    wnid2counts = {wnid:0 for wnid in wnids}
    for wnid in df['wnid']:
        wnid2counts[wnid] += 1
    
    df = df[['TEXT', 'wnid']]
    labels = [wnid_to_label[wnid] for wnid in df['wnid']]
    filepaths = ["../datasets/laionet/{}/{}.jpg".format(wnid, index) for wnid, index in zip(df['wnid'], df.index)]
    df = df.assign(label=labels, filepath=filepaths)
    df = df.rename(columns={'TEXT': 'caption'})
    df = df[['filepath', 'label', 'caption']]
    df['caption'] = df['caption'].apply(lambda x: x.replace('\r', ' ').replace('\n', ' ').replace('\t', ' '))

    print(df)

    freq_df = pd.DataFrame({'classname': classnames, 'counts': list(wnid2counts.values())})
    
    if args.n is not None:
        df.to_csv('../datasets/imagenet-captions/laionet_thresh{}_{}.tsv'.format(args.thresh, len(df)), sep='\t', index=False)
        freq_df.to_csv('../metadata/freqs/class_frequency_laionet_thresh{}_n{}_ori.txt'.format(args.thresh, args.n), sep='\t', index=False, header=False)
    elif args.frac < 1:
        df.to_csv('../datasets/imagenet-captions/laionet_thresh{}_frac{}_{}.tsv'.format(args.thresh, args.frac, len(df)), sep='\t', index=False)
        freq_df.to_csv('../metadata/freqs/class_frequency_laionet_thresh{}_frac{}_ori.txt'.format(args.thresh, args.frac), sep='\t', index=False, header=False)
    else:
        df.to_csv('../datasets/imagenet-captions/laionet_thresh{}_{}.tsv'.format(args.thresh, len(df)), sep='\t', index=False)
        freq_df.to_csv('../metadata/freqs/class_frequency_laionet_thresh{}_ori.txt'.format(args.thresh), sep='\t', index=False, header=False)
    