import os
import json
import random
import argparse
import numpy as np
import pandas as pd

from nltk.corpus import wordnet as wn
from metadata.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES

def get_synset(wnid):
    if wnid == 'n02112837':
        return wn.synsets('siberian_husky')[0]
    else:
        return wn.synset_from_pos_and_offset(wnid[0], int(wnid[1:]))

def join_caption(d, use_title=True, use_tags=True, use_description=True, append_cname=False, use_synset=False, template='{}'):
    assert use_title or use_tags or use_description or append_cname
    output = ''
    wnid, title, tags, description = d['wnid'], d['title'], d['tags'], d['description']
    if isinstance(template, list):
        template = random.choice(template)
    if use_title and title != '':
        output += title
    if use_tags and len(tags) > 0:
        if output != '':
            output += ' '
        output += ' '.join(tags)
    if use_description and description != '':
        if output != '':
            output += ' '
        output += description
    # replace breakline with space
    output = output.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    if append_cname:
        if use_synset:
            cname = random.choice(wnid2synset[wnid])
        else:
            cname = wnid2classname[wnid]

        output = template.format(cname) + ' ' + output
    return output

def dump2csv(root, use_title=True, use_tags=True, use_description=True, append_cname=False, use_synset=False, shuffle=True, mode='cname'):
    assert use_title or use_tags or use_description or append_cname

    if mode == 'cname':
        template = '{}'
    elif mode == 'a+cname':
        template = 'a {}'
    elif mode == 'photo':
        template = 'a photo of a {}'
    elif mode == 'openai80':
        template = OPENAI_IMAGENET_TEMPLATES
    else:
        raise NotImplementedError

    metadata = json.load(open(os.path.join(root, 'imagenet-captions', 'imagenet_captions.json')))
    if shuffle:
        random.seed(0)
        random.shuffle(metadata)
    filepaths = [os.path.join(root, 'imagenet', 'train', d['wnid'], d['filename']) for d in metadata if d['wnid'] in wnids]
    labels = [wnin2lab[d['wnid']] for d in metadata if d['wnid'] in wnids]
    captions = [join_caption(d, use_title, use_tags, use_description, append_cname, use_synset, template) for d in metadata if d['wnid'] in wnids]

    output = {'filepath': [], 'label': [], 'caption': []}
    for filepath, label, caption in zip(filepaths, labels, captions):
        if caption.replace(' ', '') != '':
            output['filepath'].append(filepath)
            output['label'].append(label)
            output['caption'].append(caption)
    print(len(output['filepath']))

    template_only = not (use_title or use_tags or use_description)
    save_name = '{}{}{}{}{}{}_{}.tsv'.format('incaps',
                                             '_title' if use_title else '',
                                             '_tags' if use_tags else '',
                                             '_description' if use_description else '',
                                             '_' + mode if template_only else '+' + mode if append_cname else '',
                                             '_synset' if use_synset else '',
                                             len(output['filepath']))
    # dump to csv with headers
    df = pd.DataFrame(output)
    df.to_csv(os.path.join(root, 'imagenet-captions', save_name), sep='\t', index=False)

    freq_names = ['', 'description_only', 'tags_only', 'no_title',
                  'title_only', 'no_tags', 'no_description', 'full_text']
    freq_id = (use_description << 0) + (use_tags << 1) + (use_title << 2) \
        if not append_cname else 0 # all are chosen
    freq_name = freq_names[freq_id]
    if not os.path.exists('../metadata/freqs/class_frequency_incaps_{}_imagenet_ori.txt'.format(freq_name)):
        counts = [0] * len(wnids)
        for lab in output['label']:
            counts[lab] += 1
        freqs = pd.DataFrame({'classname': classnames, 'count': counts})
        freqs.to_csv('../metadata/freqs/class_frequency_incaps_{}_imagenet_ori.txt'.format(freq_name), sep='\t', header=False, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../datasets/') # specify it to your dataset path
    parser.add_argument('--use_title', action='store_true', default=False)
    parser.add_argument('--use_tags', action='store_true', default=False)
    parser.add_argument('--use_description', action='store_true', default=False)
    parser.add_argument('--append_cname', action='store_true', default=False) # this ensures same data scale between variants
    parser.add_argument('--use_synset', action='store_true', default=False)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='cname', choices=['cname', 'a+cname', 'photo', 'openai80'])
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)

    wnids = sorted(json.load(open("../metadata/descriptors/descriptors_imagenet_synset.json", "r")).keys())
    classnames = list(json.load(open('../metadata/descriptors/descriptors_imagenet.json', 'r')).keys())
    wnid2classname = dict(zip(wnids, classnames))
    wnin2lab = {wnid: i for i, wnid in enumerate(wnids)}
    wnid2synset = {wnid: [lemma.name().replace('_', ' ') for lemma in get_synset(wnid).lemmas()] for wnid in wnids}

    dump2csv(args.root, args.use_title, args.use_tags, args.use_description, args.append_cname, args.use_synset, args.shuffle, args.mode)