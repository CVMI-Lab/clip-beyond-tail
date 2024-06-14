MASTER_PORT=$((RANDOM % 101 + 20000))
cd ./dino
torchrun --nproc_per_node 8 --master_port=$MASTER_PORT main_dino.py \
    --arch rn50 \
    --dataset imagenet \
    --data_path ../datasets/laionet \
    --output_dir ./output/dino_laionet_rn50_ep100_bs1024_sample4096 \
    --optimizer sgd \
    --batch_size_per_gpu 128 \
    --lr 0.03 \
    --weight_decay 1e-4 \
    --weight_decay_end 1e-4 \
    --global_crops_scale 0.14 1 \
    --local_crops_scale 0.05 0.14 \
    --epochs 100 \
    --num_workers 10 \
    --freeze_last_layer 0 \
    --sample_num 4096