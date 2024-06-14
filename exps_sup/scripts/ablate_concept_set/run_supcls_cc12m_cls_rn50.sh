MASTER_PORT=$((RANDOM % 101 + 20000))
torchrun --nproc_per_node 4 --master_port=$MASTER_PORT train_sup.py \
    --arch rn50 \
    --dataset cc12m-cls \
    --data_path ../datasets/cc12m/cc12m_cls.tsv \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-4 \
    --epochs 90 \
    --scheduler step \
    --val_freq 1 \
    --num_workers 8 \
    --log_dir ./output \
    --name rn50_cc12mcls_supcls_bs256 \
    --frequency_file ../metadata/freqs/class_frequency_cc12m_cls_imagenet_ori.txt \
    --imb_metrics \
    --nc_metrics