MASTER_PORT=$((RANDOM % 101 + 20000))
torchrun --nproc_per_node 4 --master_port=$MASTER_PORT train_sup.py \
    --arch rn50 \
    --dataset yfcc15m-cls \
    --data_path ../datasets/yfcc15m/yfcc15m_cls.tsv \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-4 \
    --epochs 90 \
    --scheduler step \
    --val_freq 1 \
    --num_workers 8 \
    --log_dir ./output \
    --name rn50_yfcc15mcls_supcls_bs256 \
    --frequency_file ../metadata/freqs/class_frequency_yfcc15m_cls_imagenet_ori.txt \
    --imb_metrics \
    --nc_metrics