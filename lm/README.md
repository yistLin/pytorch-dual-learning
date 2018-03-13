# Language Model

This language model is heavily depended on [Word-level language modeling RNN - pytorch/examples](https://github.com/pytorch/examples/tree/master/word_language_model). To train it, just use the code here and follow the steps provided there.

### Usage

Reload pre-trained model and dictionary first, and use `get_prob()` to get language model probability. 

```python
words = ['we', 'have', 'told', 'that', 'this', 'will']
lmprob = LMProb('wmt16-en.pt', 'data/wmt16-en/dict.pkl')
norm_prob = lmprob.get_prob(words, verbose=True)
print('norm_prob = {:.4f}'.format(norm_prob))
```
