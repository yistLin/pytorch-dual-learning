# Language Model

### Usage

Reload model and dictionary first, and use `get_prob()` to get language model probability. 

```python
words = ['we', 'have', 'told', 'that', 'this', 'will']
lmprob = LMProb('wmt16-en.pt', 'data/wmt16-en/dict.pkl')
norm_prob = lmprob.get_prob(words, verbose=True)
print('norm_prob = {:.4f}'.format(norm_prob))
```
