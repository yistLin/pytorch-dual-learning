#!/bin/bash
python3 nmt.py --mode test --test_src wmt16-data/valid.en --test_tgt wmt16-data/valid.de --load_model model.wmt16-ende.bin --save_to_file decoded2.txt
