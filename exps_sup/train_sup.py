import os
import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms

import utils
import model
from model import ClsWrapper
from dataset import ImageNetCaptionCls, CsvDataset

def get_parser():
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--arch', default='rn50', type=str, help='Architecture')
    parser.add_argument('--amp', default=False, action='store_true', help="Floating point precision.")
    parser.add_argument('--visual_weights', default='', type=str, help="Path to pretrained visual weights.")
    parser.add_argument('--head_weights', default='', type=str, help="Path to pretrained head weights.")
    parser.add_argument('--freeze_visual', action='store_true', help="Freeze visual weights.")
    parser.add_argument('--freeze_attnpool', action='store_true', help="Freeze attnpool weights.")
    parser.add_argument('--freeze_head', action='store_true', help="Freeze head weights.")
    parser.add_argument('--dataset', default='imagenet-captions', type=str, help='Dataset')
    parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N', help='mini-batch size (default: 256), this is the total '
        'batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--scheduler', default='step', type=str, help='lr scheduler (default: step)', choices=['step', 'cosine'])
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='../datasets/imagenet', type=str)
    parser.add_argument('--imagenet_path', default='../datasets/imagenet', type=str)
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--log_dir', default="./output", help='Path to save logs and checkpoints.')
    parser.add_argument('--name', default="", type=str, help='Name of experiment.')
    parser.add_argument('--frequency_file', type=str, default=None, help="Path to file with class frequencies for zero shot evaluation.")
    parser.add_argument('--imb_metrics', action='store_true', help="Compute metrics for imbalanced classification.")
    parser.add_argument('--nc_metrics', action='store_true', help="Compute metrics for neural collapse theory.")
    parser.add_argument('--num_classes', default=1000, type=int, help='Number of labels for linear classifier.')
    parser.add_argument('--output_dim', default=None, type=int, help='Dimension of the visual features.')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set.')
    parser.add_argument('--sample_head', default=False, action='store_true')
    parser.add_argument('--sample_weight', default='linear', type=str, choices=['uniform', 'linear', 'sqrt'])
    parser.add_argument('--num_to_sample', default=50, type=int)
    parser.add_argument(
        "--imagenet100_index_file",
        type=str,
        default="../metadata/imagenet100_idxs.txt",
        help="Path to file with class indexes of imagenet-100.",
    )
    args = parser.parse_args()
    return args

def sample_class_inds(gt_classes, num_to_sample, C, weight=None):
    appeared = torch.unique(gt_classes) # C'
    prob = appeared.new_ones(C).float()
    if len(appeared) < num_to_sample:
        if weight is not None:
            prob[:C] = weight.float().clone()
        prob[appeared] = 0
        more_appeared = torch.multinomial(
            prob, num_to_sample - len(appeared),
            replacement=False)
        appeared = torch.cat([appeared, more_appeared])
        appeared, _ = torch.sort(appeared)
    
    new_gt = torch.searchsorted(appeared, gt_classes)
    return appeared, new_gt

def load_weight(weight, _model, part='all', idxs=None):
    if os.path.isdir(weight):
        weight = os.path.join(weight, 'checkpoint.pth.tar')
    weights = torch.load(weight)
    if os.path.isfile(weight):
        print("=> loading weight '{}'".format(weight))
        if part == 'visual':
            checkpoint = {k.replace('module.', ''):v for k,v in weights.items()}
            msg = _model.load_state_dict(checkpoint, strict=False)
        elif part == 'head':
            if idxs is not None:
                weights = weights[idxs]
            checkpoint = {'head': weights}
            embedding_dim = checkpoint['head'].shape[1]
            if _model.visual.output_dim != embedding_dim:
                print("=> embedding dim mismatch, reconfiguring visual model")
                _model.visual = model.__dict__[_model.model_str](output_dim=embedding_dim).to(_model.device)
                _model.head = nn.Parameter(torch.empty(_model.num_classes, embedding_dim)).to(_model.device)
            msg = _model.load_state_dict(checkpoint, strict=False)
        else:
            checkpoint = {k.replace('module.', ''):v for k,v in weights['state_dict'].items()}
            if idxs is not None:
                checkpoint['head'] = checkpoint['head'][idxs]
            msg = _model.load_state_dict(checkpoint, strict=True)
        print("=> loaded weight '{}' with msg: {}".format(weight, msg))
    else:
        print("=> no weight found at '{}'".format(weight))

def main(args):
    utils.init_distributed_mode(args)

    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    if torch.cuda.is_available():
        # Copy-paste from open_clip
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    args.batch_size = int(args.batch_size / args.world_size)

    if utils.is_main_process() and not args.evaluate:
        if args.name == "":
            date_str = utils.get_timestamp()
            args.name = '_'.join([args.arch,
                                  "incaps" if args.dataset == "imagenet-captions" else "in" if args.dataset == "imagenet" else args.dataset,
                                  'supcls',
                                  'bs{}'.format(args.batch_size * args.world_size)
                                  ])
            if args.freeze_visual:
                args.name += '_freezevis'
                if args.freeze_attnpool:
                    args.name += '+attn'
                args.name += '_' + args.visual_weights.split('/')[-1].split('.')[0]
            
            if args.freeze_head:
                args.name += '_freezehead_' + args.head_weights.split('/')[-1].split('.')[0]
            
            args.name += '_' + date_str
        args.log_dir = os.path.join(args.log_dir, args.name)
        os.makedirs(args.log_dir, exist_ok=True)
        with open(os.path.join(args.log_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    # ============ building network ... ============
    model = ClsWrapper(args.arch, args.num_classes, args.output_dim)
    model.cuda()
    model.eval()

    if args.num_classes == 100:
        idxs = np.loadtxt(args.imagenet100_index_file, dtype=int).tolist()
    else:
        idxs = None

    if args.evaluate:
        load_weight(args.log_dir, model, 'all', idxs)

    if args.visual_weights:
        load_weight(args.visual_weights, model, 'visual', idxs)
    if args.head_weights:
        load_weight(args.head_weights, model, 'head', idxs)

    for name, param in model.named_parameters():
        if name.startswith('visual') and not 'attnpool' in name and args.freeze_visual:
            param.requires_grad = False
        if 'attnpool' in name and args.freeze_attnpool:
            param.requires_grad = False
        if name.startswith('head') and args.freeze_head:
            param.requires_grad = False

    print(f"Model {args.arch} built.")

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=utils.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(utils.OPENAI_DATASET_MEAN, utils.OPENAI_DATASET_STD),
    ])

    # ============ preparing data ... ============
    dataset_val = datasets.ImageFolder(os.path.join(args.imagenet_path, "val"), transform=val_transform)
    val_loader = torch.utilss.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.evaluate:
        test_stats = validate_network(model, val_loader, args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0), interpolation=utils.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(utils.OPENAI_DATASET_MEAN, utils.OPENAI_DATASET_STD),
    ])

    if args.data_path.endswith('.csv') or args.data_path.endswith('.tsv'):
        dataset_train = CsvDataset(args.data_path, transform=train_transform)
    elif args.dataset == 'imagenet-captions':
        dataset_train = ImageNetCaptionCls(args.data_path, transform=train_transform)
    else:
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)
    sampler = torch.utilss.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utilss.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # set optimizer
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr * (args.batch_size * args.world_size) / 256., # linear scaling rule
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epochs // 3, gamma=0.1)

    scaler = GradScaler() if args.amp else None

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.log_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]
        
    for epoch in range(start_epoch, args.epochs):
        args.current_epoch = epoch
        train_loader.sampler.set_epoch(epoch)

        train(model, optimizer, scaler, train_loader, epoch, args)
        scheduler.step()

        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(model, val_loader, args)
            if utils.is_main_process():
                print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
                best_acc = max(best_acc, test_stats["acc1"])
                print(f'Max accuracy so far: {best_acc:.2f}%')
        
        if utils.is_main_process():
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            if scaler is not None:
                save_dict["scaler"] = scaler.state_dict()
            torch.save(save_dict, os.path.join(args.log_dir, "checkpoint.pth.tar"))
    
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train(model, optimizer, scaler, loader, epoch, args):
    if args.sample_head and args.frequency_file is not None:
        with open(args.frequency_file) as f:
            lines = f.readlines()
        freqs = torch.Tensor([int(line.strip().split('\t')[1]) for line in lines]).cuda()
    else:
        freqs = None

    if args.num_classes == 100:
        idxs = np.loadtxt(args.imagenet100_index_file, dtype=int).tolist()
        if freqs is not None:
            freqs = freqs[idxs]
    else:
        idxs = None

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('logit_scale', utils.SmoothedValue(window_size=1, fmt='{value:.3f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for (input, target) in metric_logger.log_every(loader, 1000, header):
        # move to gpu
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with autocast(scaler is not None):
            if args.sample_head:
                if args.sample_weight == 'uniform' or freqs is None:
                    weight = None
                elif args.sample_weight == 'linear':
                    weight = freqs
                elif args.sample_weight == 'sqrt':
                    weight = torch.sqrt(freqs)
                else:
                    raise NotImplementedError
                sampled_labels, new_target = sample_class_inds(target, args.num_to_sample, args.num_classes, weight=weight)
                output, _ = model(input, sampled_labels)
                loss = nn.CrossEntropyLoss()(output, new_target)
            else:
                output, _ = model(input)
                loss = nn.CrossEntropyLoss()(output, target)

        # compute the gradients
        optimizer.zero_grad()
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            utils.unwrap_model(model).logit_scale.clamp_(0, np.log(100))

        # log
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(logit_scale=utils.unwrap_model(model).logit_scale.exp().item())
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)


@torch.no_grad()
def validate_network(model, val_loader, args):
    if not utils.is_main_process():
        return {}
    # important!!!
    model = model.module
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    C = model.num_classes if 'imagenet100' not in args.imagenet_path else 100
    all_features, all_targets = [], []
    confusion_matrix = torch.zeros(C, C).cuda()

    if not os.path.isdir(args.log_dir):
        args.log_dir = os.path.dirname(args.log_dir)

    if C == 100:
        idxs = np.loadtxt(args.imagenet100_index_file, dtype=int).tolist()
    else:
        idxs = None

    for input, target in metric_logger.log_every(val_loader, 100, header):
        # move to gpu
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output, features = model(input)
        if idxs is not None:
            output = output[:,idxs]
        loss = nn.CrossEntropyLoss()(output, target)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        batch_size = input.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        if args.nc_metrics:
            all_features.append(features)
            all_targets.append(target)

        # update confusion matrix
        _, predicted = torch.max(output, 1)
        for t, p in zip(target.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    log_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    save_metrics = {}
    if args.imb_metrics:
        per_class_accs, per_class_pred_cnt = utils.get_imb_metrics(confusion_matrix)
        save_metrics.update({'confusion':confusion_matrix.cpu(), 'per_class_accs': per_class_accs.cpu(), 'per_class_pred_cnt': per_class_pred_cnt.cpu()})
    if args.nc_metrics:
        all_features = torch.cat(all_features, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        Sw_invSb, cos_M_all, cos_W_all, cos_M_nearest, cos_W_nearest = utils.get_nc_metrics(model, all_features, all_targets)
        save_metrics.update({'Sw_invSb': Sw_invSb.cpu(), 'cos_M_all': cos_M_all.cpu(), 'cos_W_all': cos_W_all.cpu(), 'cos_M_nearest': cos_M_nearest.cpu(), 'cos_W_nearest': cos_W_nearest.cpu()})
        del all_features, all_targets

    if args.imb_metrics or args.nc_metrics:
        metrics_path = os.path.join(args.log_dir, 'metrics')
        if not os.path.exists(metrics_path):
            os.makedirs(metrics_path)
        
        try:
            freqs = np.array([int(line.strip().split('\t')[1]) for line in open(args.frequency_file, 'r')])
            if idxs is not None:
                freqs = freqs[idxs]
            corr_statistics = utils.get_corrs(save_metrics, freqs)
            log_stats.update(corr_statistics)
        except:
            print("Frequency file not found, skipping correlation statistics.")

    if args.num_classes == 100:
        log_stats = {"val-in100-" + name: val for name, val in log_stats.items()}
    else:
        log_stats = {"val-" + name: val for name, val in log_stats.items()}
    if args.evaluate and args.head_weights:
        head_name = '_' + args.head_weights.split('/')[-1].split('.')[0]
    elif args.evaluate:
        head_name = 'evalonly'
    else:
        head_name = ''
    
    if args.num_classes == 100:
        head_name += '_in100'
    
    with open(os.path.join(args.log_dir, "stats_val{}.jsonl".format(head_name)), "a+") as f:
        f.write(json.dumps(log_stats))
        f.write("\n")

    metrics_path = os.path.join(args.log_dir, 'metrics{}'.format(head_name))
    os.makedirs(metrics_path, exist_ok=True)
    if save_metrics:
        if not args.evaluate:
            torch.save(save_metrics, os.path.join(metrics_path, f"metrics_val_ep{args.current_epoch}.pt"))
        torch.save(save_metrics, os.path.join(metrics_path, f"metrics_val_latest.pt"))

    return log_stats


if __name__ == '__main__':
    args = get_parser()
    main(args)