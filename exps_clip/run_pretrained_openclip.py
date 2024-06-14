import subprocess
import shlex
import open_clip
from tqdm import tqdm

def parse_cmd(model, pretrained):
    if 'cc12m' in pretrained or 'yfcc15m' in pretrained or 'metaclip' in pretrained:
        dataset = pretrained
    elif 'laion400m' in pretrained:
        dataset = 'laion400m'
    elif 'laion2b' in pretrained:
        dataset = 'laion2b'
    else:
        raise NotImplementedError

    command = "MASTER_PORT=$((RANDOM % 101 + 20000))" + "\n" + \
        "cd open_clip/src" + "\n" + \
        "torchrun --nproc_per_node 1 --master_port=$MASTER_PORT -m training.main \\" + "\n" + \
        "\t--model {} \\".format(model) + "\n" + \
        "\t--pretrained {} \\".format(pretrained) + "\n" + \
        "\t--imagenet-val ../datasets/imagenet/val/ \\" + "\n" + \
        "\t--imagenet-v2 ../datasets/imagenetv2/ \\" + "\n" + \
        "\t--frequency-file ../metadata/freqs/class_frequency_{}_imagenet_ori.txt \\".format(dataset) + "\n" + \
        "\t--imb_metrics \\" + "\n" + \
        "\t--nc_metrics \\" + "\n" + \
        "\t--logs ./logs_pretrained \\" + "\n" + \
        "\t--name {}-{}".format(model, pretrained) + "\n"
    return command

if __name__ == "__main__":
    models = open_clip.list_pretrained()
    models = [item for item in models if 'cc12m' in item[1] or 'yfcc15m' in item[1] or 'laion400m' in item[1] or 'laion2b' in item[1] or 'metaclip' in item[1]]
    print("{} models in total: ".format(len(models)))

    for model, pretrained in tqdm(models):
        print("Processing {}-{}".format(model, pretrained))
        command = parse_cmd(model, pretrained)
        with open('temp.sh', 'w') as f:
            f.write(command)
        args = shlex.split('bash temp.sh')
        subprocess.Popen(args).wait()