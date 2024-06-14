MASTER_PORT=$((RANDOM % 101 + 20000))
cd open_clip/src
torchrun --nproc_per_node 4 --master_port=$MASTER_PORT -m training.main \
    --model RN50 \
    --train-data ../datasets/imagenet-captions/in100caps_title_tags_description_cname_44548.tsv \
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
    --num-classes 100 \
    --imagenet-100 ../datasets/imagenet100/val/ \
    --imagenet100-index-file ../datasets/imagenet-captions/metadata/imagenet100_idxs.txt \
    --frequency-file ../metadata/freqs/class_frequency_incaps_imagenet_ori.txt \
    --imb_metrics \
    --nc_metrics \
    --logs ./logs \
    --name RN50-lr_0.001-b_1024-e_32-p_amp-in100