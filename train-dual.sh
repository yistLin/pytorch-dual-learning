#!/bin/bash

nmtdir=/data/groups/chatbot/dl_data/wmt16-small
lmdir=/data/groups/chatbot/dl_data/lm
srcdir=/data/groups/chatbot/dl_data/wmt16-dual

nmtA=$nmtdir/model.wmt16-ende-small.best.bin
nmtB=$nmtdir/model.wmt16-deen-small.best.bin
lmA=$lmdir/wmt16-en.pt
lmB=$lmdir/wmt16-de.pt
lmA_dict=$lmdir/dict.en.pkl
lmB_dict=$lmdir/dict.de.pkl
srcA=$srcdir/train-small.en
srcB=$srcdir/train-small.de

saveA="modelA"
saveB="modelB"

python3 dual.py \
    --nmt $nmtA $nmtB \
    --lm $lmA $lmB \
    --dict $lmA_dict $lmB_dict \
    --src $srcA $srcB \
    --log_every 5 \
    --save_n_iter 400 \
    --alpha 0.01 \
    --model $saveA $saveB

