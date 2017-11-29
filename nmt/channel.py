# -*- coding: utf-8 -*-

import sys
import torch
import argparse

from util import read_corpus, data_iter
from model import NMT


def sample(args):
    train_data_src = read_corpus(args.src_file, source='src')
    train_data_tgt = read_corpus(args.tgt_file, source='tgt')
    train_data = zip(train_data_src, train_data_tgt)

    # load model params
    print('load model from [%s]' % args.model_bin, file=sys.stderr)
    params = torch.load(args.model_bin, map_location=lambda storage, loc: storage)
    vocab = params['vocab']
    opt = params['args']
    state_dict = params['state_dict']

    # build model
    model = NMT(opt, vocab)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.cuda()

    # sampling
    print('begin sampling')
    train_iter = cum_samples = 0
    for src_sents, tgt_sents in data_iter(train_data, batch_size=1):
        train_iter += 1
        samples = model.sample(src_sents, sample_size=5, to_word=True)
        cum_samples += sum(len(sample) for sample in samples)

        for i, tgt_sent in enumerate(tgt_sents):
            print('*' * 80)
            print('target:' + ' '.join(tgt_sent))
            tgt_samples = samples[i]
            print('samples:')
            for sid, sample in enumerate(tgt_samples, 1):
                print('[%d] %s' % (sid, ' '.join(sample[1:-1])))
            print('*' * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_bin')
    parser.add_argument('src_file')
    parser.add_argument('tgt_file')
    args = parser.parse_args()

    sample(args)

