MASTER_PORT=$((RANDOM % 101 + 20000))
torchrun --nproc_per_node 4 --master_port=$MASTER_PORT train_sup.py \
    --arch rn50 \
    --dataset imagenet-captions \
    --data_path ../datasets/imagenet-captions/incaps_title_tags_description_cname_fstail50ceil1.tsv \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-4 \
    --epochs 90 \
    --scheduler step \
    --val_freq 1 \
    --num_workers 8 \
    --log_dir ./output \
    --frequency_file ../metadata/freqs/class_frequency_incaps_fstail50ceil1_imagenet_ori.txt \
    --name rn50_incaps_supcls_bs256_fstail50ceil1 \
    --imb_metrics \
    --nc_metrics