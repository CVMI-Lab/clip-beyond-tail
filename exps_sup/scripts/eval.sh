CKPT=$1
MASTER_PORT=$((RANDOM % 101 + 20000))
torchrun --nproc_per_node 1 --master_port=$MASTER_PORT train_sup.py \
     --arch=rn50 --log_dir ${CKPT} --evaluate --imb_metrics --nc_metrics \
     --frequency_file metadata/freqs/class_frequency_incaps_imagenet_ori.txt