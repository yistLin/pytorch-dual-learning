###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import os
import math
import pickle
import argparse
import numpy as np

import torch
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wmt16',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model-wmt16.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='output.txt',
                    help='output file for generated text')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()
model.cuda()

with open(os.path.join(args.data, 'dict.pkl'), 'rb') as f:
    dictionary = pickle.load(f)

# corpus = data.Corpus(args.data)
# ntokens = len(corpus.dictionary)
ntokens = len(dictionary)
hidden = model.init_hidden(1)

# Test sentence
# words = ['<sos>', 'we', 'have', 'told', 'that', 'this', 'will']
words = ['<sos>', 'we', 'should', 'be', 'ensuring', 'that', 'the', 'principle', 'of', 'the']
indxs = [dictionary.getid(x) for x in words]

# Test random
# sos_tag = np.array([corpus.dictionary.word2idx['<sos>']]).astype(int)
# indxs = np.concatenate((sos_tag, np.random.randint(ntokens, size=6)), axis=0)
# words = [corpus.dictionary.idx2word[x] for x in indxs]

input = Variable(torch.LongTensor([int(indxs[0])]).unsqueeze(0), volatile=True)
input.data = input.data.cuda()

print('words =', words)
print('indxs =', indxs)

sum_prob = 0.0

for i in range(1, len(words)):
    output, hidden = model(input, hidden)
    word_weights = output.squeeze().data.exp().cpu()

    prob = word_weights[indxs[i]] / word_weights.sum()
    log_prob = math.log(prob)
    sum_prob += log_prob
    input.data.fill_(int(indxs[i]))
    print('  {}\t=> {:d}, P(w|s)={:.4f}, sum_prob={:.2f}'.format(words[i], indxs[i], prob, sum_prob))

norm_prob = sum_prob / (len(words) - 1)
print('\n  => sum_prob / {:d} = {:.4f}'.format(len(words)-1, norm_prob))

