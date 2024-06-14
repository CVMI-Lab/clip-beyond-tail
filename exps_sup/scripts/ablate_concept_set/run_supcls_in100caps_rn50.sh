MASTER_PORT=$((RANDOM % 101 + 20000))
torchrun --nproc_per_node 4 --master_port=$MASTER_PORT train_sup.py \
    --arch rn50 \
    --dataset in100cap \
    --data_path ../datasets/imagenet-captions/in100caps_cname_44548.tsv \
    --imagenet_path ../datasets/imagenet100 \
    --num_classes 100 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-4 \
    --epochs 90 \
    --scheduler step \
    --val_freq 1 \
    --num_workers 8 \
    --log_dir ./output \
    --frequency_file ../metadata/freqs/class_frequency_incaps_imagenet_ori.txt \
    --imb_metrics \
    --nc_metrics