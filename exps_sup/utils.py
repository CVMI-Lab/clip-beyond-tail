"""
Misc functions.

Some are copy-paste from torchvision references or other public repos like DETR and DINO:
https://github.com/facebookresearch/dino/blob/main/utils.py
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
import time
import datetime
from collections import defaultdict, deque
from scipy.stats import spearmanr
import torch
import torch.distributed as dist
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def get_imb_metrics(confusion):
    per_class_accs = confusion.diag() / confusion.sum(1)
    per_class_pred_cnt = confusion.sum(0)
    return per_class_accs, per_class_pred_cnt


@torch.no_grad()
def get_nc_metrics(model, features, targets):
    C = model.num_classes
    D = model.visual.output_dim
    mean_features = torch.zeros(C, D, device=features.device)
    num_samples = torch.zeros(C, device=features.device)
    Sw = torch.zeros(C, D, D, device=features.device)
    for computation in ['Mean','Cov']:
        for c in range(C):
            idxs = (targets == c).nonzero(as_tuple=True)[0]
            if len(idxs) == 0: # If no class-c in this batch
                continue
            h_c = features[idxs,:]

            if computation == 'Mean':
                mean_features[c,:] += torch.sum(h_c, dim=0)
                num_samples[c] += h_c.shape[0]
            elif computation == 'Cov':
                z = h_c - mean_features[c].unsqueeze(0) # B D
                cov = torch.matmul(z.unsqueeze(-1), z.unsqueeze(1)) # B D D
                Sw[c,:,:] += torch.sum(cov, dim=0) # D D
        if computation == 'Mean':
            mean_features /= num_samples.unsqueeze(-1)
        elif computation == 'Cov':
            Sw /= num_samples.sum()
        
    # global mean
    global_means = torch.mean(mean_features, dim=0, keepdim=True) # 1 D
    # between-class covariance
    centered_means = mean_features - global_means
    Sb = torch.matmul(centered_means.T, centered_means) / C

    # avg norm
    prototypes = model.head
    M_norms = torch.norm(centered_means.T, dim=0)
    W_norms = torch.norm(prototypes.T, dim=0)

    # tr{Sw Sb^-1}
    invSb = torch.linalg.pinv(Sb)
    Sw_invSb = torch.matmul(Sw, invSb).diagonal(dim1=-2, dim2=-1).sum(-1)

    cos_M_all, cos_M_nearest_all = mutual_coherence(centered_means.T/M_norms)
    cos_W_all, cos_W_nearest_all = mutual_coherence(prototypes.T/W_norms)
    return Sw_invSb, cos_M_all, cos_W_all, cos_M_nearest_all, cos_W_nearest_all


def mutual_coherence(V):
    C = V.shape[1]
    G = V.T @ V
    G += 1 / (C-1)
    G -= torch.diag(torch.diag(G))
    margins = G.abs().sum(dim=1) / (C-1)
    margins_nearest = G.abs().max(dim=1)[0]
    return margins, margins_nearest


def get_corrs(metrics, freqs):
    res = {}
    try:
        accs, preds = metrics['per_class_accs'].numpy(), metrics['per_class_pred_cnt'].numpy()
        corr_acc, corr_pred = spearmanr(freqs, accs).statistic, spearmanr(freqs, preds).statistic
        res['corr_acc'], res['corr_pred'] = corr_acc, corr_pred
    except:
        # print('Imbalance metrics are not supported')
        pass
    try:
        Sw_invSb, cos_M_all, cos_W_all, cos_M_nearest, cos_W_nearest \
            = metrics['Sw_invSb'].numpy(), metrics['cos_M_all'].numpy(), metrics['cos_W_all'].numpy(), metrics['cos_M_nearest'].numpy(), metrics['cos_W_nearest'].numpy()
        corr_Sw_invSb, corr_cos_M_all, corr_cos_W_all, corr_cos_M_nearest, corr_cos_W_nearest \
            = spearmanr(freqs, Sw_invSb).statistic, spearmanr(freqs, cos_M_all).statistic, \
            spearmanr(freqs, cos_W_all).statistic, spearmanr(freqs, cos_M_nearest).statistic, spearmanr(freqs, cos_W_nearest).statistic
        corr_Sw_invSb_acc, corr_cos_M_all_acc, corr_cos_W_all_acc, corr_cos_M_nearest_acc, corr_cos_W_nearest_acc \
            = spearmanr(accs, Sw_invSb).statistic, spearmanr(accs, cos_M_all).statistic, spearmanr(accs, cos_W_all).statistic, \
                spearmanr(accs, cos_M_nearest).statistic, spearmanr(accs, cos_W_nearest).statistic
        res['corr_Sw_invSb'], res['corr_cos_M_all'], res['corr_cos_W_all'], res['corr_cos_M_nearest'], res['corr_cos_W_nearest'] \
            = corr_Sw_invSb, corr_cos_M_all, corr_cos_W_all, corr_cos_M_nearest, corr_cos_W_nearest
        res['corr_Sw_invSb_acc'], res['corr_cos_M_all_acc'], res['corr_cos_W_all_acc'], res['corr_cos_M_nearest_acc'], res['corr_cos_W_nearest_acc'] \
            = corr_Sw_invSb_acc, corr_cos_M_all_acc, corr_cos_W_all_acc, corr_cos_M_nearest_acc, corr_cos_W_nearest_acc
    except:
        # print('NC metrics are not supported')
        pass
    return res