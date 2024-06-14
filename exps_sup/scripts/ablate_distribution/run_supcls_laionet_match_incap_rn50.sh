MASTER_PORT=$((RANDOM % 101 + 20000))
torchrun --nproc_per_node 4 --master_port=$MASTER_PORT train_sup.py \
    --arch rn50 \
    --dataset laionet_match_incap \
    --data_path ../datasets/imagenet-captions/laionet_match_incaps_416489.tsv \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-4 \
    --epochs 90 \
    --scheduler step \
    --val_freq 1 \
    --num_workers 8 \
    --log_dir ./output \
    --frequency_file ../metadata/freqs/class_frequency_laionet_match_incaps_ori.txt \
    --imb_metrics \
    --nc_metrics