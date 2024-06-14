CKPT=$1
TEMPLATE=$2
MASTER_PORT=$((RANDOM % 101 + 20000))
DATASETS=../datasets
cd open_clip/src
torchrun --nproc_per_node 1 --master_port=$MASTER_PORT -m training.main \
    --model RN50 \
    --resume latest \
    --imagenet-val ${DATASETS}/imagenet/val/ \
    --imagenet-v2 ${DATASETS}/imagenetv2/ \
    --frequency-file ../metadata/freqs/class_frequency_incaps_imagenet_ori.txt \
    --imb_metrics \
    --nc_metrics \
    --template-type ${TEMPLATE} \
    --logs ./logs \
    --name ${CKPT}