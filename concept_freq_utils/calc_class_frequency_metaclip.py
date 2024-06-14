import argparse

import re
import json

from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
class_frequencies = Counter()

def preprocess_text(text):
    if text is None:
        return []
    # Tokenize the text into individual words
    tokens = word_tokenize(re.sub(r'[^a-zA-Z ]', ' ', text.lower()).replace("'", ''))
    # Lemmatize the tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return set(tokens)

def preprocess_template(template, dataset='imagenet'):
    if isinstance(template, str):
        # Make it lower-cased
        template = template.lower()
        # Remove (xxx) in text
        template = re.sub(r'\([^)]*\)', '', template)
        # Separate template into subgroups by 'or', and reduce spaces
        template = [_t.strip() for _t in re.split(' or | / ', template)]
    else:
        template = [t.lower() for t in template]
        template = [re.sub(r'\([^)]*\)', '', t) for t in template]
    # Negative words
    neg_template = []
    if dataset == 'imagenet':
        if 'sorrel' in template:
            neg_template = ['plant', 'herb', 'flower', 'leaf']
        elif 'horizontal bar' in template:
            neg_template = ['chart', 'plot', 'graph', 'diagram']
        elif 'impala' in template:
            neg_template = ['car', 'automobile', 'vehicle', 'chevy']
        elif 'bow' in template:
            neg_template = ['tie']
        elif 'ringlet' in template:
            neg_template = ['hair']
        elif 'ram' in template:
            neg_template = ['car', 'truck', 'automobile', 'vehicle', 'dodge', 'logo', 'computer', 'memory', 'random access', 'chip', 'review']
        elif 'crane' in template:
            neg_template = ['bird', 'fish', 'water', 'wing', 'leg', 'zoo']
        elif 'sub' in template:
            neg_template = ['sandwich', 'bread', 'italian', 'meatball', 'grill', 'menu']
        elif 'sidewinder' in template:
            neg_template = ['missile', 'army', 'military', 'resort', 'park']
    elif dataset == 'places365':
        if 'arcade' in template:
            template = ['arcade passageway', 'arcade walkway', 'arcade hallway', 'arcade corridor']
        elif 'lock_chamber' in template:
            template = ['lock chamber canal', 'lock chamber waterway', 'lock chamber water']
    elif dataset == 'cub':
        if 'cardinal' in template:
            template = ['cardinal bird', 'northern cardinal', 'red cardinal']
    # Tokenize the text into individual words
    template = [word_tokenize(re.sub(r'[^a-zA-Z ]', ' ', _t).replace("'", '')) for _t in template]
    neg_template = [word_tokenize(re.sub(r'[^a-zA-Z ]', ' ', _t).replace("'", '')) for _t in neg_template]
    # Lemmatize the tokens
    template = [[lemmatizer.lemmatize(token) for token in _t] for _t in template]
    neg_template = [[lemmatizer.lemmatize(token) for token in _t] for _t in neg_template]
    return template, neg_template

def calc_frequency(template):
    cnt = 0
    t, neg_t = template
    for token, count in zip(tokens, counts):
        # Check existance in one text
        negflag = False
        for _nt in neg_t:
            if any(_w in token for _w in _nt):
                negflag = True
                break
        
        if negflag:
            continue

        for _t in t:
            if all(_w in token for _w in _t):
                cnt += count
                break
    return cnt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate word frequency from Parquet files")
    parser.add_argument("--json_path", default='../datasets/MetaCLIP/metaclip/datacard_400m.json', help="Path to metadata")
    parser.add_argument("--dataset", default='imagenet', help="Dataset name")

    args = parser.parse_args()

    # Preprocess templates
    print("Preprocessing templates...")
    class_names = list(json.load(open("../metadata/descriptors/descriptors_{}.json".format(args.dataset), "r")).keys()) # list of strings
    if 'imagenet' in args.dataset:
        class_names_in = list(json.load(open("../metadata/descriptors/descriptors_imagenet_synset.json", "r")).values()) # list of lists
        templates = [preprocess_template(class_name, args.dataset) for class_name in class_names_in]
    else:
        templates = [preprocess_template(class_name, args.dataset) for class_name in class_names]

    print("Example templates:")
    print(templates[:5])

    print("Loading metadata...")
    with open(args.json_path, "r") as f:
        metadata = json.load(f)
    
    texts = list(metadata.keys())
    counts = list(metadata.values())

    print("Preprocessing text data...")
    # Process text data
    with Pool(processes=16) as pool:
        tokens = list(tqdm(pool.imap(preprocess_text, texts), total=len(texts)))

    print("Example tokens:")
    print(tokens[:5])

    print("Calculating class frequency...")
    with Pool(processes=16) as pool:
        frequencies = list(tqdm(pool.imap(calc_frequency, templates), total=len(templates)))
    
    class_frequencies.update(dict(zip(class_names, frequencies)))

    # Dump class frequency to file
    class_frequencies_sorted = sorted(class_frequencies.items(), key=lambda x: x[1], reverse=True)
    with open("../metadata/freqs/class_frequency_{}_{}.txt".format(args.json_path.split('/')[-1].split('.')[0].replace('datacard', 'metaclip').replace('fullcc2', 'fullcc'), args.dataset), "w") as f:
        for class_name, count in class_frequencies_sorted:
            f.write(f"{class_name}\t{count}\n")
    
    with open("../metadata/freqs/class_frequency_{}_{}_ori.txt".format(args.json_path.split('/')[-1].split('.')[0].replace('datacard', 'metaclip').replace('fullcc2', 'fullcc'), args.dataset), "w") as f:
        for class_name, count in class_frequencies.items():
            f.write(f"{class_name}\t{count}\n")