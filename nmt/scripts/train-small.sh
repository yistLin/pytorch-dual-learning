#!/bin/sh

L1=$1
L2=$2
JOB=$3

data_dir="./wmt16-small-data"
vocab_bin="$data_dir/vocab.$L1$L2.bin"
train_src="$data_dir/train.$L1"
train_tgt="$data_dir/train.$L2"
dev_src="$data_dir/valid.$L1"
dev_tgt="$data_dir/valid.$L2"
test_src="$data_dir/test.$L1"
test_tgt="$data_dir/test.$L2"

job_name="$JOB"
model_name="model.${job_name}"

python3 nmt.py \
    --cuda \
    --mode train \
    --vocab ${vocab_bin} \
    --save_to ${model_name} \
    --log_every 50 \
    --valid_niter 2500 \
    --valid_metric ppl \
    --save_model_after 2 \
    --beam_size 5 \
    --batch_size 64 \
    --hidden_size 256 \
    --embed_size 256 \
    --uniform_init 0.1 \
    --dropout 0.2 \
    --clip_grad 5.0 \
    --lr_decay 0.5 \
    --train_src ${train_src} \
    --train_tgt ${train_tgt} \
    --dev_src ${dev_src} \
    --dev_tgt ${dev_tgt}

