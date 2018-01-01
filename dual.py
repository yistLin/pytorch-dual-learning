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
        model_id = (['A', 'B'])[i]
        print('loading pieces, part {:s}'.format(model_id))

        print('  load model{:s}     from [{:s}]'.format(model_id, args.nmt[i]), file=sys.stderr)
        params = torch.load(args.nmt[i], map_location=lambda storage, loc: storage)  # load model onto CPU
        vocabs[model_id] = params['vocab']
        opts[model_id] = params['args']
        state_dicts[model_id] = params['state_dict']

        print('  load train_src{:s} from [{:s}]'.format(model_id, args.src[i]), file=sys.stderr)
        train_srcs[model_id] = read_corpus(args.src[i], source='src')

        print('  load lm{:s}        from [{:s}]'.format(model_id, args.lm[i]), file=sys.stderr)
        lms[model_id] = LMProb(args.lm[i], args.dict[i])

    models = {}
    optimizers = {}

    for m in ['A', 'B']:
        # build model
        opts[m].cuda = args.cuda

        models[m] = NMT(opts[m], vocabs[m])
        models[m].load_state_dict(state_dicts[m])
        models[m].train()

        if args.cuda:
            models[m] = models[m].cuda()

        random.shuffle(train_srcs[m])

        # optimizer
        # optimizers[m] = torch.optim.Adam(models[m].parameters())
        optimizers[m] = torch.optim.SGD(models[m].parameters(), lr=1e-3, momentum=0.9)

    # loss function
    loss_nll = torch.nn.NLLLoss()
    loss_ce = torch.nn.CrossEntropyLoss()

    epoch = 0
    start = args.start_iter

    while True:
        epoch += 1
        print('\nstart of epoch {:d}'.format(epoch))

        data = {}
        data['A'] = iter(train_srcs['A'])
        data['B'] = iter(train_srcs['B'])

        start += (epoch - 1) * len(train_srcs['A']) + 1

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

                    var_ids = Variable(torch.LongTensor(ids[1:]), requires_grad=False)
                    NLL_losses.append(loss_nll(dist, var_ids).cpu())

                    lm_probs.append(lmB.get_prob(smid))

                    src_sent_var = to_input_variable([smid], vocabB.src, cuda=args.cuda)
                    tgt_sent_var = to_input_variable([['<s>'] + s + ['</s>']], vocabB.tgt, cuda=args.cuda)
                    src_sent_len = [len(smid)]

                    score = modelB(src_sent_var, src_sent_len, tgt_sent_var[:-1]).squeeze(1)

                    CE_losses.append(loss_ce(score, tgt_sent_var[1:].view(-1)).cpu())

                # losses on target language
                fw_losses = torch.cat(NLL_losses)

                # losses on reconstruction
                bw_losses = torch.cat(CE_losses)

                # r1, language model reward
                r1s = Variable(torch.FloatTensor(lm_probs), requires_grad=False)
                r1s = (r1s - torch.mean(r1s)) / torch.std(r1s)

                # r2, communication reward
                r2s = Variable(bw_losses.data, requires_grad=False)
                r2s = (torch.mean(r2s) - r2s) / torch.std(r2s)

                # rk = alpha * r1 + (1 - alpha) * r2
                rks = r1s * args.alpha + r2s * (1 - args.alpha)

                # averaging loss over samples
                A_loss = torch.mean(fw_losses * rks)
                B_loss = torch.mean(bw_losses * (1 - args.alpha))

                if show_log:
                    for r1, r2, rk, fw_loss, bw_loss in zip(r1s.data.numpy(), r2s.data.numpy(), rks.data.numpy(), fw_losses.data.numpy(), bw_losses.data.numpy()):
                        print('r1={:7.4f}\t r2={:7.4f}\t rk={:7.4f}\t fw_loss={:7.4f}\t bw_loss={:7.4f}'.format(r1, r2, rk, fw_loss, bw_loss))
                    print('A loss = {:.7f} \t B loss = {:.7f}'.format(A_loss.data.numpy().item(), B_loss.data.numpy().item()))

                optimizerA.zero_grad()
                optimizerB.zero_grad()

                A_loss.backward()
                B_loss.backward()

                optimizerA.step()
                optimizerB.step()

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
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    print(args)

    dual(args)

