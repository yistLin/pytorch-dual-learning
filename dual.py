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
    for i in range(2):
        mid = (['A', 'B'])[i]
        print('loading pieces, part {:s}'.format(mid))

        print('  load model{:s}     from [{:s}]'.format(mid, args.nmt[i]), file=sys.stderr)
        params = torch.load(args.nmt[i], map_location=lambda storage, loc: storage)
        vocabs[mid] = params['vocab']
        opts[mid] = params['args']
        state_dicts[mid] = params['state_dict']

        print('  load train_src{:s} from [{:s}]'.format(mid, args.src[i]), file=sys.stderr)
        train_srcs[mid] = read_corpus(args.src[i], source='src')

        print('  load lm{:s}        from [{:s}]'.format(mid, args.lm[i]), file=sys.stderr)
        lms[mid] = LMProb(args.lm[i], args.dict[i])

    models = {}
    optimizers = {}

    for m in ['A', 'B']:
        # build model
        if args.cuda:
            opts[m].cuda = True
        else:
            opts[m].cuda = False

        models[m] = NMT(opts[m], vocabs[m])
        models[m].load_state_dict(state_dicts[m])
        models[m].train()

        if args.cuda:
            models[m] = models[m].cuda()

        random.shuffle(train_srcs[m])

        # optimizer
        optimizers[m] = torch.optim.Adam(models[m].parameters())

    # loss function
    loss_nll = torch.nn.NLLLoss()
    loss_ce = torch.nn.CrossEntropyLoss()

    epoch = 0
    while True:
        epoch += 1
        print('\nstart of epoch {:d}'.format(epoch))

        data = {}
        data['A'] = iter(train_srcs['A'])
        data['B'] = iter(train_srcs['B'])

        start = (epoch - 1) * len(train_srcs['A']) + 1
        for t in range(start, start + len(train_srcs['A'])):
            show_log = False
            if t % args.log_every == 0:
                show_log = True
            
            if show_log:
                print('\nstep', t)

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

                if show_log:
                    print('\n{:s} -> {:s}'.format(m, change(m)))
                    print('[s]', ' '.join(s))

                hyps = modelA.beam(s, beam_size=5)

                for ids, smid, dist in hyps:
                    if show_log:
                        print('[smid]', ' '.join(smid))

                    var_ids = torch.autograd.Variable(torch.LongTensor(ids[1:]), requires_grad=False)
                    NLL_losses.append(loss_nll(dist, var_ids).cpu())

                    lm_probs.append(lmB.get_prob(smid))

                    src_sent_var = to_input_variable([smid], vocabB.src, cuda=args.cuda)
                    tgt_sent_var = to_input_variable([['<s>'] + s + ['</s>']], vocabB.tgt, cuda=args.cuda)
                    src_sent_len = [len(smid)]

                    score = modelB(src_sent_var, src_sent_len, tgt_sent_var[:-1]).squeeze(1)

                    CE_losses.append(loss_ce(score, tgt_sent_var[1:].view(-1)).cpu())

                r1_mean = sum(lm_probs) / len(lm_probs)
                r1 = [Variable(torch.FloatTensor([p - r1_mean]), requires_grad=False) for p in lm_probs]

                r2_mean = sum(CE_losses) / len(CE_losses)
                r2 = [Variable(-(l.data - r2_mean.data), requires_grad=False) for l in CE_losses]

                if show_log:
                    for a, b, in zip(r1, r2):
                        print('r1 = {:.4f} \t r2 = {:.4f}'.format(a.data.numpy().item(), b.data.numpy().item()))

                alpha = Variable(torch.FloatTensor([args.alpha]), requires_grad=False)
                beta = Variable(torch.FloatTensor([1 - args.alpha]), requires_grad=False)
                rk = [a * alpha + b * beta for a, b in zip(r1, r2)]

                optimizerA.zero_grad()
                optimizerB.zero_grad()

                A_loss = torch.mean(torch.cat(NLL_losses) * torch.cat(rk))
                B_loss = torch.mean(torch.cat(CE_losses)) * beta
                
                A_loss.backward()
                B_loss.backward()

                optimizerA.step()
                optimizerB.step()

                if show_log:
                    print('A loss = {:.7f} \t B loss = {:.7f}'.format(A_loss.data.numpy().item(), B_loss.data.numpy().item()))
            
            if t % args.save_n_iter == 0:
                print('\nsaving model')
                models['A'].save('{}.iter{}.bin'.format(args.model[0], t))
                models['B'].save('{}.iter{}.bin'.format(args.model[1], t))


def change(m):
    if m == 'A':
        return 'B'
    else:
        return 'A'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nmt', nargs=2, required=True, help='pre-train nmt model path')
    parser.add_argument('--lm', nargs=2, required=True, help='language model path')
    parser.add_argument('--dict', nargs=2, required=True, help='dictionary path')
    parser.add_argument('--src', nargs=2, required=True, help='training data path')
    parser.add_argument('--model', nargs=2, type=str, default=['modelA', 'modelB'])
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--save_n_iter', type=int, default=1000)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    dual(args)

