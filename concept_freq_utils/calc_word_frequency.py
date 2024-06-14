import argparse
import glob
import os
import re
import pickle
import pandas
import pyarrow.parquet as pq

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

def preprocess_template(template):
    # Make it lower-cased
    template = template.lower()
    # Separate template into subgroups by 'or', and reduce spaces
    template = [_t.strip() for _t in template.split('or')]
    # Tokenize the text into individual words
    template = [word_tokenize(re.sub(r'[^a-zA-Z ]', ' ', _t).replace("'", '')) for _t in template]
    # Lemmatize the tokens
    template = [[lemmatizer.lemmatize(token) for token in _t] for _t in template]
    return template

def calc_frequency(all_tokens, template):
    count = 0
    for tokens in all_tokens:
        # Check existance in one text
        for _t in template:
            if all(_w in tokens for _w in _t):
                count += 1
                break
    return count

def check_exist(templates, tokens):
    is_exist = []
    for idx, template in enumerate(templates):
        # Check existance in one text
        for _t in template:
            if all(_w in tokens for _w in _t):
                is_exist.append(idx)
                break
    return is_exist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate word frequency from Parquet files")
    parser.add_argument("--url_path", default='../datasets/laion/laion400m-meta', help="Folder containing Parquet files")
    parser.add_argument("--input_format", default='parquet', help="Supported Parquet file format")
    parser.add_argument("--caption_col", default='TEXT', help="Column containing the text/captions")

    args = parser.parse_args()

    # Read Parquet files
    file_paths = sorted(glob.glob(args.url_path + "/*." + args.input_format))
    
    if not os.path.exists("tokens"):
        os.makedirs("tokens")
    for i, file_path in enumerate(file_paths):
        print("Processing {}/{} files".format(i + 1, len(file_paths)))
        # Try to reuse processed tokens
        if os.path.isfile("tokens/tokens_{}.pkl".format(file_path.split('/')[-1].split('.')[0])):
            print("Loading tokens from file...")
            with open("tokens/tokens_{}.pkl".format(file_path.split('/')[-1].split('.')[0]), "rb") as f:
                tokens = pickle.load(f)
        else:
            if args.input_format == 'parquet':
                df = pq.read_table(file_path).to_pandas()
            else:
                df = pandas.read_csv(file_path, sep='\t')
            text_data = df[args.caption_col].tolist()
            del df
            print("Preprocessing text data...")
            # Process text data
            with Pool(processes=16) as pool:
                tokens = list(tqdm(pool.imap(preprocess_text, text_data), total=len(text_data)))
            # Dump tokens to file (warning: these files can be very large)
            with open("tokens/tokens_{}.pkl".format(file_path.split('/')[-1].split('.')[0]), "wb") as f:
                pickle.dump(tokens, f)
        
        print("Example tokens:")
        print(tokens[:5])

        print("Calculating word frequency...")
        for token in tqdm(tokens):
            class_frequencies.update(token)

    # Dump word frequency to file
    class_frequencies = sorted(class_frequencies.items(), key=lambda x: x[1], reverse=True)
    with open("../metadata/freqs/word_frequency_{}.txt".format(args.url_path.split('/')[-1]), "w") as f:
        for class_name, count in class_frequencies:
            f.write(f"{class_name}\t{count}\n")