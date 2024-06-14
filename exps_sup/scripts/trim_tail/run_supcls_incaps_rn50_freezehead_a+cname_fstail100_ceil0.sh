MASTER_PORT=$((RANDOM % 101 + 20000))
torchrun --nproc_per_node 4 --master_port=$MASTER_PORT train_sup.py \
    --arch rn50 \
    --dataset imagenet-captions \
    --data_path ../datasets/imagenet-captions/incaps_title_tags_description_cname_fstail100ceil0.tsv \
    --freeze_head \
    --head_weights ../metadata/heads/in1k_clip_rn50_wit400m_a+cname.pt \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-4 \
    --epochs 90 \
    --scheduler step \
    --val_freq 1 \
    --num_workers 8 \
    --log_dir ./output \
    --frequency_file ../metadata/freqs/class_frequency_incaps_fstail100ceil0_imagenet_ori.txt \
    --name rn50_incaps_supcls_bs256_fstail100ceil0_freezehead_in1k_clip_rn50_wit400m_a+cname \
    --imb_metrics \
    --nc_metrics