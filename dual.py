# -*- coding: utf-8 -*-

import sys
import torch
import argparse
import random

from torch.autograd import Variable

from nmt import read_corpus, data_iter
from nmt import NMT, to_input_variable

from lm import LMProb
from lm import model

def dual(args):
    vocabs = {}
    opts = {}
    state_dicts = {}
    train_srcs = {}
    lms = {}

    # load model params & training data
    print('load modelA from [{:s}]'.format(args.modelA_bin), file=sys.stderr)
    params = torch.load(args.modelA_bin, map_location=lambda storage, loc: storage)
    vocabs['A'] = params['vocab']
    opts['A'] = params['args']
    state_dicts['A'] = params['state_dict']
    print('load train_srcA from [{:s}]'.format(args.train_srcA), file=sys.stderr)
    train_srcs['A'] = read_corpus(args.train_srcA, source='src')
    print('load lmA from [{:s}]'.format(args.lmA), file=sys.stderr)
    lms['A'] = LMProb(args.lmA, args.lmAdict)

    print('load modelB from [{:s}]'.format(args.modelB_bin), file=sys.stderr)
    params = torch.load(args.modelB_bin, map_location=lambda storage, loc: storage)
    vocabs['B'] = params['vocab']
    opts['B'] = params['args']
    state_dicts['B'] = params['state_dict']    
    print('load train_srcB from [{:s}]'.format(args.train_srcB), file=sys.stderr)
    train_srcs['B'] = read_corpus(args.train_srcB, source='src')
    print('load lmB from [{:s}]'.format(args.lmB), file=sys.stderr)
    lms['B'] = LMProb(args.lmB, args.lmBdict)

    models = {}
    optimizers = {}

    for m in ['A', 'B']:
        # build model
        models[m] = NMT(opts[m], vocabs[m])
        models[m].load_state_dict(state_dicts[m])
        models[m].train()
        models[m] = models[m].cuda()

        random.shuffle(train_srcs[m])

        # optimizer
        optimizers[m] = torch.optim.Adam(models[m].parameters())

    # loss function
    loss_nll = torch.nn.NLLLoss()
    loss_ce = torch.nn.CrossEntropyLoss()

    epoch = 0
    while True:
        if epoch == 2:
            break
        epoch += 1
        print('start of epoch {:d}'.format(epoch))

        data = {}
        data['A'] = iter(train_srcs['A'])
        data['B'] = iter(train_srcs['B'])

        for t in range(0, len(train_srcs['A'])):
            print('sent', t)
            for m in ['A', 'B']:
                lm_probs = []

                NLL_losses = []
                CE_losses = []

                modelA = models[m]
                modelB = models[change(m)]
                lmB = lms[change(m)]
                optimizerA = optimizers[m]
                optimizerB = optimizers[change(m)]
                vocabB = vocabs[change(m)]
                s = next(data[m])

                hyps = modelA.beam(s, beam_size=5)

                for ids, smid, dist in hyps:
                    var_ids = torch.autograd.Variable(torch.LongTensor(ids[1:]), requires_grad=False)
                    NLL_losses.append(loss_nll(dist, var_ids).cpu())

                    lm_probs.append(lmB.get_prob(smid))

                    src_sent_var = to_input_variable([smid], vocabB.src, cuda=True)
                    tgt_sent_var = to_input_variable([['<s>'] + s + ['</s>']], vocabB.tgt, cuda=True)
                    src_sent_len = [len(smid)]

                    score = modelB(src_sent_var, src_sent_len, tgt_sent_var[:-1]).squeeze(1)

                    CE_losses.append(loss_ce(score, tgt_sent_var[1:].view(-1)).cpu())

                r1_mean = sum(lm_probs) / len(lm_probs)
                r1 = [Variable(torch.FloatTensor([p - r1_mean]), requires_grad=False) for p in lm_probs]

                r2_mean = sum(CE_losses) / len(CE_losses)
                r2 = [Variable(-(l.data - r2_mean.data), requires_grad=False) for l in CE_losses]

                rk = [a + b for a, b in zip(r1, r2)]

                optimizerA.zero_grad()
                optimizerB.zero_grad()

                torch.mean(torch.cat(NLL_losses) * torch.cat(rk)).backward()
                torch.mean(torch.cat(CE_losses)).backward()

                optimizerA.step()
                optimizerB.step()


def change(m):
    if m == 'A':
        return 'B'
    else:
        return 'A'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('modelA_bin')
    parser.add_argument('modelB_bin')
    parser.add_argument('lmA')
    parser.add_argument('lmAdict')
    parser.add_argument('lmB')
    parser.add_argument('lmBdict')
    parser.add_argument('train_srcA')
    parser.add_argument('train_srcB')
    args = parser.parse_args()

    dual(args)

