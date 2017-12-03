# -*- coding: utf-8 -*-
import sys

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


def input_transpose(sents, pad_token):
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    masks = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])
        masks.append([1 if len(sents[k]) > i else 0 for k in range(batch_size)])

    return sents_t, masks


def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]


def tensor_transform(linear, X):
    # X is a 3D tensor
    return linear(X.contiguous().view(-1, X.size(2))).view(X.size(0), X.size(1), -1)


class NMT(nn.Module):
    def __init__(self, args, vocab):
        super(NMT, self).__init__()

        self.args = args

        self.vocab = vocab

        self.src_embed = nn.Embedding(len(vocab.src), self.args.embed_size, padding_idx=vocab.src['<pad>'])
        self.tgt_embed = nn.Embedding(len(vocab.tgt), self.args.embed_size, padding_idx=vocab.tgt['<pad>'])

        self.encoder_lstm = nn.LSTM(self.args.embed_size, self.args.hidden_size, bidirectional=True, dropout=self.args.dropout)
        self.decoder_lstm = nn.LSTMCell(self.args.embed_size + self.args.hidden_size, self.args.hidden_size)

        # attention: dot product attention
        # project source encoding to decoder rnn's h space
        self.att_src_linear = nn.Linear(self.args.hidden_size * 2, self.args.hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)
        self.att_vec_linear = nn.Linear(self.args.hidden_size * 2 + self.args.hidden_size, self.args.hidden_size, bias=False)

        # prediction layer of the target vocabulary
        self.readout = nn.Linear(self.args.hidden_size, len(vocab.tgt), bias=False)

        # dropout layer
        self.dropout = nn.Dropout(self.args.dropout)

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(self.args.hidden_size * 2, self.args.hidden_size)

    def forward(self, src_sents, src_sents_len, tgt_words):
        src_encodings, init_ctx_vec = self.encode(src_sents, src_sents_len)
        scores = self.decode(src_encodings, init_ctx_vec, tgt_words)

        return scores

    def encode(self, src_sents, src_sents_len):
        """
        :param src_sents: (src_sent_len, batch_size), sorted by the length of the source
        :param src_sents_len: (src_sent_len)
        """
        # (src_sent_len, batch_size, embed_size)
        src_word_embed = self.src_embed(src_sents)
        packed_src_embed = pack_padded_sequence(src_word_embed, src_sents_len)

        # output: (src_sent_len, batch_size, hidden_size)
        output, (last_state, last_cell) = self.encoder_lstm(packed_src_embed)
        output, _ = pad_packed_sequence(output)

        dec_init_cell = self.decoder_cell_init(torch.cat([last_cell[0], last_cell[1]], 1))
        dec_init_state = F.tanh(dec_init_cell)

        return output, (dec_init_state, dec_init_cell)

    def decode(self, src_encoding, dec_init_vec, tgt_sents):
        """
        :param src_encoding: (src_sent_len, batch_size, hidden_size)
        :param dec_init_vec: (batch_size, hidden_size)
        :param tgt_sents: (tgt_sent_len, batch_size)
        :return:
        """
        init_state = dec_init_vec[0]
        init_cell = dec_init_vec[1]
        hidden = (init_state, init_cell)

        new_tensor = init_cell.data.new
        batch_size = src_encoding.size(1)

        # (batch_size, src_sent_len, hidden_size * 2)
        src_encoding = src_encoding.permute(1, 0, 2)
        # (batch_size, src_sent_len, hidden_size)
        src_encoding_att_linear = tensor_transform(self.att_src_linear, src_encoding)
        # initialize attentional vector
        att_tm1 = Variable(new_tensor(batch_size, self.args.hidden_size).zero_(), requires_grad=False)

        tgt_word_embed = self.tgt_embed(tgt_sents)
        scores = []

        # start from `<s>`, until y_{T-1}
        for y_tm1_embed in tgt_word_embed.split(split_size=1):
            # input feeding: concate y_tm1 and previous attentional vector
            x = torch.cat([y_tm1_embed.squeeze(0), att_tm1], 1)

            # h_t: (batch_size, hidden_size)
            h_t, cell_t = self.decoder_lstm(x, hidden)
            h_t = self.dropout(h_t)

            ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encoding, src_encoding_att_linear)

            att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))   # E.q. (5)
            att_t = self.dropout(att_t)

            score_t = self.readout(att_t)   # E.q. (6)
            scores.append(score_t)

            att_tm1 = att_t
            hidden = h_t, cell_t

        scores = torch.stack(scores)
        return scores

    def translate(self, src_sents, beam_size=None, to_word=True):
        """
        perform beam search
        TODO: batched beam search
        """
        if not type(src_sents[0]) == list:
            src_sents = [src_sents]
        if not beam_size:
            beam_size = self.args.beam_size

        src_sents_var = to_input_variable(src_sents, self.vocab.src, cuda=self.args.cuda, is_test=True)

        src_encoding, dec_init_vec = self.encode(src_sents_var, [len(src_sents[0])])
        src_encoding_att_linear = tensor_transform(self.att_src_linear, src_encoding)

        init_state = dec_init_vec[0]
        init_cell = dec_init_vec[1]
        hidden = (init_state, init_cell)

        att_tm1 = Variable(torch.zeros(1, self.args.hidden_size), volatile=True)
        hyp_scores = Variable(torch.zeros(1), volatile=True)
        if self.args.cuda:
            att_tm1 = att_tm1.cuda()
            hyp_scores = hyp_scores.cuda()

        eos_id = self.vocab.tgt['</s>']
        bos_id = self.vocab.tgt['<s>']
        tgt_vocab_size = len(self.vocab.tgt)

        hypotheses = [[bos_id]]
        completed_hypotheses = []
        completed_hypothesis_scores = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < self.args.decode_max_time_step:
            t += 1
            hyp_num = len(hypotheses)

            expanded_src_encoding = src_encoding.expand(src_encoding.size(0), hyp_num, src_encoding.size(2))
            expanded_src_encoding_att_linear = src_encoding_att_linear.expand(src_encoding_att_linear.size(0), hyp_num, src_encoding_att_linear.size(2))

            y_tm1 = Variable(torch.LongTensor([hyp[-1] for hyp in hypotheses]), volatile=True)
            if self.args.cuda:
                y_tm1 = y_tm1.cuda()

            y_tm1_embed = self.tgt_embed(y_tm1)

            x = torch.cat([y_tm1_embed, att_tm1], 1)

            # h_t: (hyp_num, hidden_size)
            h_t, cell_t = self.decoder_lstm(x, hidden)
            h_t = self.dropout(h_t)

            ctx_t, alpha_t = self.dot_prod_attention(h_t, expanded_src_encoding.permute(1, 0, 2), expanded_src_encoding_att_linear.permute(1, 0, 2))

            att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))
            att_t = self.dropout(att_t)

            score_t = self.readout(att_t)
            p_t = F.log_softmax(score_t)

            live_hyp_num = beam_size - len(completed_hypotheses)
            new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(p_t) + p_t).view(-1)
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=live_hyp_num)
            prev_hyp_ids = top_new_hyp_pos / tgt_vocab_size
            word_ids = top_new_hyp_pos % tgt_vocab_size
            # new_hyp_scores = new_hyp_scores[top_new_hyp_pos.data]

            new_hypotheses = []

            live_hyp_ids = []
            new_hyp_scores = []
            for prev_hyp_id, word_id, new_hyp_score in zip(prev_hyp_ids.cpu().data, word_ids.cpu().data, top_new_hyp_scores.cpu().data):
                hyp_tgt_words = hypotheses[prev_hyp_id] + [word_id]
                if word_id == eos_id:
                    completed_hypotheses.append(hyp_tgt_words)
                    completed_hypothesis_scores.append(new_hyp_score)
                else:
                    new_hypotheses.append(hyp_tgt_words)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.LongTensor(live_hyp_ids)
            if self.args.cuda:
                live_hyp_ids = live_hyp_ids.cuda()

            hidden = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hyp_scores = Variable(torch.FloatTensor(new_hyp_scores), volatile=True) # new_hyp_scores[live_hyp_ids]
            if self.args.cuda:
                hyp_scores = hyp_scores.cuda()
            hypotheses = new_hypotheses

        if len(completed_hypotheses) == 0:
            completed_hypotheses = [hypotheses[0]]
            completed_hypothesis_scores = [0.0]

        if to_word:
            for i, hyp in enumerate(completed_hypotheses):
                completed_hypotheses[i] = [self.vocab.tgt.id2word[w] for w in hyp]

        ranked_hypotheses = sorted(zip(completed_hypotheses, completed_hypothesis_scores), key=lambda x: x[1], reverse=True)

        return [hyp for hyp, score in ranked_hypotheses]

    def sample(self, src_sents, sample_size=None, to_word=False):
        if not type(src_sents[0]) == list:
            src_sents = [src_sents]
        if not sample_size:
            sample_size = self.args.sample_size

        src_sents_num = len(src_sents)
        batch_size = src_sents_num * sample_size

        src_sents_var = to_input_variable(src_sents, self.vocab.src, cuda=self.args.cuda, is_test=True)
        src_encoding, (dec_init_state, dec_init_cell) = self.encode(src_sents_var, [len(s) for s in src_sents])

        dec_init_state = dec_init_state.repeat(sample_size, 1)
        dec_init_cell = dec_init_cell.repeat(sample_size, 1)
        hidden = (dec_init_state, dec_init_cell)

        src_encoding = src_encoding.repeat(1, sample_size, 1)
        src_encoding_att_linear = tensor_transform(self.att_src_linear, src_encoding)
        src_encoding = src_encoding.permute(1, 0, 2)
        src_encoding_att_linear = src_encoding_att_linear.permute(1, 0, 2)

        new_tensor = dec_init_state.data.new
        att_tm1 = Variable(new_tensor(batch_size, self.args.hidden_size).zero_(), volatile=True)
        y_0 = Variable(torch.LongTensor([self.vocab.tgt['<s>'] for _ in range(batch_size)]), volatile=True)

        eos = self.vocab.tgt['</s>']
        # eos_batch = torch.LongTensor([eos] * batch_size)
        sample_ends = torch.ByteTensor([0] * batch_size)
        all_ones = torch.ByteTensor([1] * batch_size)
        if self.args.cuda:
            y_0 = y_0.cuda()
            sample_ends = sample_ends.cuda()
            all_ones = all_ones.cuda()

        samples = [y_0]

        t = 0
        while t < self.args.decode_max_time_step:
            t += 1

            # (sample_size)
            y_tm1 = samples[-1]

            y_tm1_embed = self.tgt_embed(y_tm1)

            x = torch.cat([y_tm1_embed, att_tm1], 1)

            # h_t: (batch_size, hidden_size)
            h_t, cell_t = self.decoder_lstm(x, hidden)
            h_t = self.dropout(h_t)

            ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encoding, src_encoding_att_linear)

            att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
            att_t = self.dropout(att_t)

            score_t = self.readout(att_t)  # E.q. (6)
            p_t = F.softmax(score_t)

            if self.args.sample_method == 'random':
                y_t = torch.multinomial(p_t, num_samples=1).squeeze(1)
            elif self.args.sample_method == 'greedy':
                _, y_t = torch.topk(p_t, k=1, dim=1)
                y_t = y_t.squeeze(1)

            samples.append(y_t)

            sample_ends |= torch.eq(y_t, eos).byte().data
            if torch.equal(sample_ends, all_ones):
                break

            # if torch.equal(y_t.data, eos_batch):
            #     break

            att_tm1 = att_t
            hidden = h_t, cell_t

        # post-processing
        completed_samples = [list([list() for _ in range(sample_size)]) for _ in range(src_sents_num)]
        for y_t in samples:
            for i, sampled_word in enumerate(y_t.cpu().data):
                src_sent_id = i % src_sents_num
                sample_id = i // src_sents_num
                if len(completed_samples[src_sent_id][sample_id]) == 0 or completed_samples[src_sent_id][sample_id][-1] != eos:
                    completed_samples[src_sent_id][sample_id].append(sampled_word)

        if to_word:
            for i, src_sent_samples in enumerate(completed_samples):
                completed_samples[i] = word2id(src_sent_samples, self.vocab.tgt.id2word)

        return completed_samples

    def beam(self, src_sents, beam_size=3):
        """
        perform beam search
        """
        if not type(src_sents[0]) == list:
            src_sents = [src_sents]

        src_sents_var = to_input_variable(src_sents, self.vocab.src, cuda=self.args.cuda, is_test=False)

        src_encoding, dec_init_vec = self.encode(src_sents_var, [len(src_sents[0])])
        src_encoding_att_linear = tensor_transform(self.att_src_linear, src_encoding)

        init_state = dec_init_vec[0]
        init_cell = dec_init_vec[1]
        hidden = (init_state, init_cell)

        att_tm1 = Variable(torch.zeros(1, self.args.hidden_size), requires_grad=False)
        hyp_scores = Variable(torch.zeros(1), requires_grad=False)
        if self.args.cuda:
            att_tm1 = att_tm1.cuda()
            hyp_scores = hyp_scores.cuda()

        eos_id = self.vocab.tgt['</s>']
        bos_id = self.vocab.tgt['<s>']
        tgt_vocab_size = len(self.vocab.tgt)

        # store output distributions
        out_dists = [[]]
        completed_out_dists = []

        hypotheses = [[bos_id]]
        completed_hypotheses = []
        completed_hypothesis_scores = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < self.args.decode_max_time_step:
            t += 1
            hyp_num = len(hypotheses)

            expanded_src_encoding = src_encoding.expand(src_encoding.size(0), hyp_num, src_encoding.size(2))
            expanded_src_encoding_att_linear = src_encoding_att_linear.expand(src_encoding_att_linear.size(0), hyp_num, src_encoding_att_linear.size(2))

            y_tm1 = Variable(torch.LongTensor([hyp[-1] for hyp in hypotheses]), requires_grad=False)
            if self.args.cuda:
                y_tm1 = y_tm1.cuda()

            y_tm1_embed = self.tgt_embed(y_tm1)

            x = torch.cat([y_tm1_embed, att_tm1], 1)

            # h_t: (hyp_num, hidden_size)
            h_t, cell_t = self.decoder_lstm(x, hidden)
            h_t = self.dropout(h_t)

            ctx_t, alpha_t = self.dot_prod_attention(h_t, expanded_src_encoding.permute(1, 0, 2), expanded_src_encoding_att_linear.permute(1, 0, 2))

            att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))
            att_t = self.dropout(att_t)

            score_t = self.readout(att_t)
            p_t = F.log_softmax(score_t)

            live_hyp_num = beam_size - len(completed_hypotheses)
            new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(p_t) + p_t).view(-1)
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=live_hyp_num)
            prev_hyp_ids = top_new_hyp_pos / tgt_vocab_size
            word_ids = top_new_hyp_pos % tgt_vocab_size
            # new_hyp_scores = new_hyp_scores[top_new_hyp_pos.data]

            # get output distributions
            p_t_cpu = p_t.cpu()

            new_out_dists = []
            new_hypotheses = []

            live_hyp_ids = []
            new_hyp_scores = []
            for prev_hyp_id, word_id, new_hyp_score in zip(prev_hyp_ids.cpu().data, word_ids.cpu().data, top_new_hyp_scores.cpu().data):
                tgt_dists = out_dists[prev_hyp_id] + [p_t_cpu[prev_hyp_id].unsqueeze(0)]
                hyp_tgt_words = hypotheses[prev_hyp_id] + [word_id]
                if word_id == eos_id:
                    completed_out_dists.append(tgt_dists)
                    completed_hypotheses.append(hyp_tgt_words)
                    completed_hypothesis_scores.append(new_hyp_score)
                else:
                    new_out_dists.append(tgt_dists)
                    new_hypotheses.append(hyp_tgt_words)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.LongTensor(live_hyp_ids)
            if self.args.cuda:
                live_hyp_ids = live_hyp_ids.cuda()

            hidden = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hyp_scores = Variable(torch.FloatTensor(new_hyp_scores), requires_grad=False) # new_hyp_scores[live_hyp_ids]
            if self.args.cuda:
                hyp_scores = hyp_scores.cuda()

            out_dists = new_out_dists
            hypotheses = new_hypotheses

        if len(completed_hypotheses) == 0:
            completed_out_dists = [out_dists[0]]
            completed_hypotheses = [hypotheses[0]]
            completed_hypothesis_scores = [0.0]

        # convert to words
        completed_hypotheses_words = []
        for i, hyp in enumerate(completed_hypotheses):
            completed_hypotheses_words.append([self.vocab.tgt.id2word[w] for w in hyp])

        # merge variables
        for i, dists in enumerate(completed_out_dists):
            completed_out_dists[i] = torch.cat(dists, 0)

        # sort with scores
        ranked_hypotheses = sorted(zip(completed_hypotheses, completed_hypothesis_scores, completed_hypotheses_words, completed_out_dists), key=lambda x: x[1], reverse=True)

        return [(hyp, words, dist) for hyp, score, words, dist in ranked_hypotheses]

    def attention(self, h_t, src_encoding, src_linear_for_att):
        # (1, batch_size, attention_size) + (src_sent_len, batch_size, attention_size) =>
        # (src_sent_len, batch_size, attention_size)
        att_hidden = F.tanh(self.att_h_linear(h_t).unsqueeze(0).expand_as(src_linear_for_att) + src_linear_for_att)

        # (batch_size, src_sent_len)
        att_weights = F.softmax(tensor_transform(self.att_vec_linear, att_hidden).squeeze(2).permute(1, 0))

        # (batch_size, hidden_size * 2)
        ctx_vec = torch.bmm(src_encoding.permute(1, 2, 0), att_weights.unsqueeze(2)).squeeze(2)

        return ctx_vec, att_weights

    def dot_prod_attention(self, h_t, src_encoding, src_encoding_att_linear, mask=None):
        """
        :param h_t: (batch_size, hidden_size)
        :param src_encoding: (batch_size, src_sent_len, hidden_size * 2)
        :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
        :param mask: (batch_size, src_sent_len)
        """
        # (batch_size, src_sent_len)
        att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
        if mask:
            att_weight.data.masked_fill_(mask, -float('inf'))
        att_weight = F.softmax(att_weight)

        att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
        ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)

        return ctx_vec, att_weight

    def save(self, path):
        print('save parameters to [%s]' % path, file=sys.stderr)
        params = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)


def to_input_variable(sents, vocab, cuda=False, is_test=False):
    """
    return a tensor of shape (src_sent_len, batch_size)
    """

    word_ids = word2id(sents, vocab)
    sents_t, masks = input_transpose(word_ids, vocab['<pad>'])

    sents_var = Variable(torch.LongTensor(sents_t), volatile=is_test, requires_grad=False)
    if cuda:
        sents_var = sents_var.cuda()

    return sents_var

