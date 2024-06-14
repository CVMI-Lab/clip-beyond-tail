CKPT=$1
DATASET=$2
python ./ssl-transfer/linear.py -d ${DATASET} -n openai -m ${CKPT}
# python ./ssl-transfer/linear.py -d food       -n openai -m ${CKPT}
# python ./ssl-transfer/linear.py -d cifar10    -n openai -m ${CKPT}
# python ./ssl-transfer/linear.py -d cifar100   -n openai -m ${CKPT}
# python ./ssl-transfer/linear.py -d birdsnap   -n openai -m ${CKPT}
# python ./ssl-transfer/linear.py -d sun397     -n openai -m ${CKPT}
# python ./ssl-transfer/linear.py -d cars       -n openai -m ${CKPT}
# python ./ssl-transfer/linear.py -d aircraft   -n openai -m ${CKPT}
# python ./ssl-transfer/linear.py -d voc2007    -n openai -m ${CKPT}
# python ./ssl-transfer/linear.py -d dtd        -n openai -m ${CKPT}
# python ./ssl-transfer/linear.py -d pets       -n openai -m ${CKPT}
# python ./ssl-transfer/linear.py -d caltech101 -n openai -m ${CKPT}
# python ./ssl-transfer/linear.py -d flowers    -n openai -m ${CKPT}