import json
import pandas as pd
from tqdm import tqdm

from nltk.corpus import wordnet as wn
from multiprocessing import Pool

import warnings
warnings.filterwarnings("ignore", 'This pattern is interpreted as a regular expression')

def get_synset(wnid):
    if wnid == 'n02112837':
        return wn.synsets('siberian_husky')[0]
    else:
        return wn.synset_from_pos_and_offset(wnid[0], int(wnid[1:]))

wnids = sorted(json.load(open("../metadata/descriptors/descriptors_imagenet_synset.json", "r")).keys())
classnames = list(json.load(open('../metadata/descriptors/descriptors_imagenet.json', 'r')).keys())
wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}
wnid2synset = {wnid: [lemma.name().replace('_', ' ') for lemma in get_synset(wnid).lemmas()] for wnid in wnids}

# path to the yfcc15m/cc12m metadata that inculdes image paths and paired captions, in our case it is a jsonl file
# data = pd.read_json(path_or_buf='../datasets/cc12m/cc12m.jsonl', lines=True)
data = pd.read_json(path_or_buf='../datasets/yfcc15m/yfcc15m.jsonl', lines=True)
data = data[['filepath', 'caption']]

def process_wnid(wnid):
    print('Processing [{}/{}]: {}'.format(wnids.index(wnid), len(wnids), wnid), flush=True)
    synset = wnid2synset[wnid]
    pattern = '\\b(' + '|'.join(synset) + ')\\b'
    idxs = data.index[data['caption'].str.contains(pattern, case=False)].tolist()
    print('Found {} images for synset {}'.format(len(idxs), wnid), flush=True)
    return set(idxs)


with Pool(16) as pool:
    all_idxs = list(tqdm(pool.imap(process_wnid, wnids), total=len(wnids)))

wnid2idxs = {wnid: idxs for wnid, idxs in zip(wnids, all_idxs)}

all_idxs, counts = [], []
for wnid in wnids:
    all_idxs += list(wnid2idxs[wnid])
    counts += [wnid_to_label[wnid]] * len(wnid2idxs[wnid])

data = data.iloc[all_idxs]
data['label'] = counts

data['caption'] = data['caption'].apply(lambda x: x.replace('\r', ' ').replace('\n', ' ').replace('\t', ' '))
data.to_csv('../datasets/yfcc15m/yfcc15m_cls.tsv', sep='\t', index=False)

freq_df = pd.DataFrame({'classname': classnames, 'counts': counts})
freq_df.to_csv('../metadata/freqs/class_frequency_yfcc15m_cls_imagenet_ori.txt', sep='\t', index=False, header=False)