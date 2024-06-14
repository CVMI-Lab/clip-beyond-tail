# Modified from: https://github.com/facebookresearch/Detic/blob/main/tools/dump_clip_features.py
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json
import torch
import itertools
from metadata.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES, SIMPLE_IMAGENET_TEMPLATES

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', default='metadata/descriptors/descriptors_imagenet.json')
    parser.add_argument('--out_path', default='metadata/heads/in1k_clip_rn50_wit400m_a+cname.pt')
    parser.add_argument('--prompt', default='a')
    parser.add_argument('--model', default='clip')
    parser.add_argument('--clip_model', default="RN50")
    parser.add_argument('--fix_space', action='store_true')
    parser.add_argument('--use_underscore', action='store_true')
    parser.add_argument('--avg_synonyms', action='store_true')
    args = parser.parse_args()

    print('Loading', args.ann)
    data = json.load(open(args.ann, 'r'))
    cat_names = list(data.keys())
    if 'imagenet' in args.ann:
        synonym_data = json.load(open(args.ann.replace('imagenet', 'imagenet_synset'), 'r'))
        synonyms = list(synonym_data.values())
    else:
        synonyms = []
    if args.fix_space:
        cat_names = [x.replace('_', ' ') for x in cat_names]
    if args.use_underscore:
        cat_names = [x.strip().replace('/ ', '/').replace(' ', '_') for x in cat_names]
    print('cat_names', cat_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.prompt == 'a':
        sentences = ['a ' + x for x in cat_names]
        sentences_synonyms = [['a ' + xx for xx in x] for x in synonyms]
    if args.prompt == 'none':
        sentences = [x for x in cat_names]
        sentences_synonyms = [[xx for xx in x] for x in synonyms]
    elif args.prompt == 'photo':
        sentences = ['a photo of a {}'.format(x) for x in cat_names]
        sentences_synonyms = [['a photo of a {}'.format(xx) for xx in x] \
            for x in synonyms]
    elif args.prompt == 'scene':
        sentences = ['a photo of a {} in the scene'.format(x) for x in cat_names]
        sentences_synonyms = [['a photo of a {} in the scene'.format(xx) for xx in x] \
            for x in synonyms]
    elif args.prompt == 'openai':
        assert args.avg_synonyms == False
        templates = OPENAI_IMAGENET_TEMPLATES
        sentences = [template(x) for x in cat_names for template in templates]
    elif args.prompt == 'simple':
        assert args.avg_synonyms == False
        templates = SIMPLE_IMAGENET_TEMPLATES
        sentences = [template(x) for x in cat_names for template in templates]

    if args.avg_synonyms:
        print('sentences_synonyms', len(sentences_synonyms), \
            sum(len(x) for x in sentences_synonyms))
    
    if 'clip' in args.model:
        if args.model == 'clip':
            import clip
            model, preprocess = clip.load(args.clip_model, device=device)
        else:
            import open_clip as clip
            model, _, preprocess = clip.create_model_and_transforms(
                args.clip_model.split('.')[0], pretrained=args.clip_model.split('.')[1], device=device)
        print('Loading CLIP')
        if args.avg_synonyms:
            sentences = list(itertools.chain.from_iterable(sentences_synonyms))
            print('flattened_sentences', len(sentences))
        text = clip.tokenize(sentences).to(device)
        with torch.no_grad():
            if len(text) > 10000:
                text_features = torch.cat([
                    model.encode_text(text[:len(text) // 2]),
                    model.encode_text(text[len(text) // 2:])],
                    dim=0)
            else:
                text_features = model.encode_text(text)
        if args.prompt in ['openai', 'simple']:
            text_features = text_features.reshape(len(cat_names), len(templates), -1).mean(dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        print('text_features.shape', text_features.shape)
        if args.avg_synonyms:
            synonyms_per_cat = [len(x) for x in sentences_synonyms]
            text_features = text_features.split(synonyms_per_cat, dim=0)
            text_features = [x.mean(dim=0) for x in text_features]
            text_features = torch.stack(text_features, dim=0)
            print('after stack', text_features.shape)
        text_features = text_features.cpu() # .numpy()

    elif args.model in ['bert', 'roberta']:
        from transformers import AutoTokenizer, AutoModel
        if args.model == 'bert':
            model_name = 'bert-large-uncased' 
        if args.model == 'roberta':
            model_name = 'roberta-large' 
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        if args.avg_synonyms:
            sentences = list(itertools.chain.from_iterable(sentences_synonyms))
            print('flattened_sentences', len(sentences))
        inputs = tokenizer(sentences, padding=True, return_tensors="pt")
        with torch.no_grad():
            model_outputs = model(**inputs)
            outputs = model_outputs.pooler_output
        text_features = outputs.detach().cpu()
        if args.avg_synonyms:
            synonyms_per_cat = [len(x) for x in sentences_synonyms]
            text_features = text_features.split(synonyms_per_cat, dim=0)
            text_features = [x.mean(dim=0) for x in text_features]
            text_features = torch.stack(text_features, dim=0)
            print('after stack', text_features.shape)
        # text_features = text_features.numpy()
        print('text_features.shape', text_features.shape)
    else:
        assert 0, args.model
    if args.out_path != '':
        print('saveing to', args.out_path)
        torch.save(text_features, args.out_path)