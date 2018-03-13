# PyTorch Dual Learning

This is the PyTorch implementation for [Dual Learning for Machine Translation](https://arxiv.org/abs/1611.00179).

The NMT models used as channels are heavily depend on [pcyin/pytorch\_nmt](https://github.com/pcyin/pytorch_nmt).

### Usage

You shall prepare these models for dual learning step:
- Language Models x 2
- Translation Models x 2

##### Warm-up Step

- Language Models \
    Check here [lm/](https://github.com/yistLin/pytorch-dual-learning/tree/master/lm)
- Translation Models \
    Check here [nmt/](https://github.com/yistLin/pytorch-dual-learning/tree/master/nmt)

##### Dual Learning Step

During the reinforcement learning process, it will gain rewards from language models and translation models, and update the translation models. \
You can find more details in the paper.

- Training \
    You can simply use this [script](https://github.com/yistLin/pytorch-dual-learning/blob/master/train-dual.sh),
 you have to modify the path and name to your models.
- Test \
    To use the trained models, you can just treat it as [NMT models](https://github.com/pcyin/pytorch_nmt).


### Test (Basic)

Firstly, we trained our basic model with 450K bilingual pair, which is only 10% data, as warm-start. Then, we set up a dual-learning game, and trained two models using reinforcement technique.

##### Configs

- Reward
    - language model reward: average over square rooted length of string
    - final reward:
        ```
        rk = 0.01 x r1 + 0.99 x r2
        ```

- Optimizer
    ```
    torch.optim.SGD(models[m].parameters(), lr=1e-3, momentum=0.9)
    ```

##### Results

- English-Deutsch
    - after 600 iterations
        ```
        BLEU = 21.39, 49.1/26.8/17.6/12.2
        ```
    - after 1200 iterations
        ```
        BLEU = 21.49, 48.6/26.6/17.4/12.0
        ```

- Deutsch-English
    - after 600 iterations
        ```
        BLEU = 25.89, 56.0/32.8/22.3/15.8
        ```
    - after 1200 iterations
        ```
        BLEU = 25.94, 55.9/32.7/22.2/15.8
        ```

##### Comparisons

| Model        | Original | iter300 | iter600 | iter900 | iter1200 | iter1500 | iter3000 | iter4500 | iter6600 |
|--------------|---------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|---------:|
| EN-DE        | 20.54    | 21.27   | 21.39   | 21.49   | 21.46    | 21.49    | 21.56    | 21.62    | 21.60    |
| EN-DE (bleu) |          | 21.42   | 21.57   | 21.55   | 21.55    |          |          |          |          |
| DE-EN        | 24.69    | 25.90   | 25.89   | 25.91   | 26.03    | 25.94    | 26.02    | 26.18    | 26.20    |
| DE-EN (bleu) |          | 25.96   | 26.25   | 26.22   | 26.18    |          |          |          |          |
