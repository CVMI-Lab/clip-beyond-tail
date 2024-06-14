import argparse
import fsspec
import torch
import torchvision.datasets as datasets
from torchvision import transforms as transforms

from pathlib import Path
from tqdm import tqdm

from sup_exps.dataset import ClsWrapper
from sup_exps.utils import BICUBIC, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from clip_exps.open_clip import create_model_and_transforms
from clip_exps.open_clip import get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES, SIMPLE_IMAGENET_TEMPLATES, CNAME_TEMPLATES, A_CNAME_TEMPLATES, PHOTO_TEMPLATES

def pt_load(file_path, map_location=None):
    if file_path.startswith('s3'):
        print('Loading remote checkpoint, which may take a bit.')
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location)
    return out


def get_features(model, dataloader, args):
    all_features, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device)
            target = target.to(args.device)
            
            if args.model[0].isupper():
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
            else:
                image_features = model.encode_image(images)
            all_features.append(image_features)
            all_targets.append(target)
    all_features, all_targets = torch.cat(all_features), torch.cat(all_targets)
    features_by_target = torch.stack([all_features[all_targets == i] for i in range(args.num_classes)]).cpu()
    return features_by_target

def get_classifier(model, template_type, args, tokenizer=None):
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    print('Building zero-shot classifier')
    if template_type == 'cname':
        templates = CNAME_TEMPLATES
    elif template_type == 'a+cname':
        templates = A_CNAME_TEMPLATES
    elif template_type == 'photo':
        templates = PHOTO_TEMPLATES
    elif template_type == 'simple':
        templates = SIMPLE_IMAGENET_TEMPLATES
    else:
        templates = OPENAI_IMAGENET_TEMPLATES
    classifier = build_zero_shot_classifier(
        model,
        tokenizer=tokenizer,
        classnames=IMAGENET_CLASSNAMES,
        templates=templates,
        num_classes_per_batch=10,
        device=args.device,
        use_tqdm=True,
    )
    return classifier.T.cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP feature extraction")
    parser.add_argument("--data-path", type=str, default="./datasets/imagenet/val", help="path to imagenet val folder")
    parser.add_argument("--model", type=str, default="RN50", help="Name of the vision backbone to use.")
    parser.add_argument("--resume", type=str, required=False, help="Resume from checkpoint.")

    parser.add_argument("--pretrained", type=str, default='', help="Use a pretrained CLIP model weights with the specified tag or file path.")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes of the dataset.")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers per GPU.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size per GPU.")
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.model[0].isupper(): # is from openclip
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            args.model,
            args.pretrained,
            device=args.device,
            output_dict=True,
        )
    else:
        model = ClsWrapper(args.model, args.num_classes)
        model.to(args.device)
        preprocess_val = transforms.Compose([
            transforms.Resize(256, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
        ])

    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            msg = model.load_state_dict(sd)
            print(msg)
            print(f"=> resuming checkpoint '{args.resume}' (epoch {epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            msg = model.load_state_dict(checkpoint)
            print(msg)
            print(f"=> loaded checkpoint '{args.resume}'")

    dataset = datasets.ImageFolder(args.data_path, transform=preprocess_val)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=None,
    )

    features = get_features(model, dataloader, args)
    if args.model[0].isupper():
        classifiers = {template: get_classifier(model, template, args) for template in ['openai', 'simple', 'cname', 'a+cname', 'photo']}
    else:
        classifiers = {'head': model.head.cpu()}
    save_dict = {'features': features, 'classifiers': classifiers}
    if args.model[0].isupper():
        save_path = './clip_exps/logs_pretrained/{}-{}/imagenet_val_features.pt'.format(args.model, args.pretrained)
    else:
        save_path = Path(args.resume).parent / 'imagenet_val_features.pt'
    torch.save(save_dict, save_path)