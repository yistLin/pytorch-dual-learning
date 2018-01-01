# -*- coding: utf-8 -*-

import math

import torch
import pickle
import numpy as np
from torch.autograd import Variable


class LMProb():

    def __init__(self, model_path, dict_path):
        with open(model_path, 'rb') as f:
            self.model = torch.load(f)
            self.model.eval()
            self.model = self.model.cpu()

        with open(dict_path, 'rb') as f:
            self.dictionary = pickle.load(f)

    def get_prob(self, words, verbose=False):
        pad_words = ['<sos>'] + words + ['<eos>']
        indxs = [self.dictionary.getid(w) for w in pad_words]
        input = Variable(torch.LongTensor([int(indxs[0])]).unsqueeze(0), volatile=True)

        if verbose:
            print('words =', pad_words)
            print('indxs =', indxs)

        hidden = self.model.init_hidden(1)
        log_probs = []
        for i in range(1, len(pad_words)):
            output, hidden = self.model(input, hidden)
            word_weights = output.squeeze().data.exp()

            prob = word_weights[indxs[i]] / word_weights.sum()
            log_probs.append(math.log(prob))
            input.data.fill_(int(indxs[i]))

        if verbose:
            for i in range(len(log_probs)):
                print('  {} => {:d},\tlogP(w|s)={:.4f}'.format(pad_words[i+1], indxs[i+1], log_probs[i]))
            print('\n  => sum_prob = {:.4f}'.format(sum(log_probs)))

        return sum(log_probs) / math.sqrt(len(log_probs))


if __name__ == '__main__':
    words = ['we', 'have', 'told', 'that', 'this', 'will']
    lmprob = LMProb('wmt16-en.pt', 'data/wmt16-en/dict.pkl')
    norm_prob = lmprob.get_prob(words, verbose=True)
    print('\n  => norm_prob = {:.4f}'.format(norm_prob))

