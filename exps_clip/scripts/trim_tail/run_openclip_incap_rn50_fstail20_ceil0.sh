MASTER_PORT=$((RANDOM % 101 + 20000))
cd open_clip/src
torchrun --nproc_per_node 4 --master_port=$MASTER_PORT -m training.main \
    --model RN50 \
    --train-data ../datasets/imagenet-captions/incaps_title_tags_description_cname_fstail20ceil0.tsv \
    --dataset-type csv \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --csv-separator '\t' \
    --batch-size 256 \
    --lr 0.001 \
    --warmup 500 \
    --epochs 32 \
    --save-frequency 8 \
    --zeroshot-frequency 1 \
    --precision amp \
    --use-bn-sync \
    --workers 8 \
    --imagenet-val ../datasets/imagenet/val/ \
    --imagenet-v2 ../datasets/imagenetv2/ \
    --frequency-file ../metadata/freqs/class_frequency_incaps_fstail20ceil0_imagenet_ori.txt \
    --imb_metrics \
    --nc_metrics \
    --logs ./logs \
    --name RN50-lr_0.001-b_1024-e_32-p_amp-fstail20ceil0