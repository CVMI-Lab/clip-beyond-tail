CKPT=$1
MASTER_PORT=$((RANDOM % 101 + 20000))
cd open_clip/src
torchrun --nproc_per_node 1 --master_port=$MASTER_PORT -m training.main \
    --model RN50 \
    --resume latest \
    --num-classes 100 \
    --imagenet-100 ../datasets/imagenet100/val/ \
    --imagenet100-index-file ../metadata/imagenet100_idxs.txt \
    --frequency-file ../metadata/freqs/class_frequency_incaps_imagenet_ori.txt \
    --imb_metrics \
    --nc_metrics \
    --logs ./logs \
    --name ${CKPT}