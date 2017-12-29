# PyTorch Dual Learning

This is the PyTorch implementation for [Dual Learning for Machine Translation](https://arxiv.org/abs/1611.00179).

The NMT models used as channels are heavily depend on [pcyin/pytorch\_nmt](https://github.com/pcyin/pytorch_nmt).

### Test (Basic)

Firstly, we trained our basic model with 450K bilingual pair, which is only 10% data, as warm-start. Then, we set up a dual-learning game, and trained two models using reinforcement technique.

##### Configs

- Reward
    ```
    rk = 0.01 x r1 + 0.99 x r2
    ```

- Optimizer
    ```
    torch.optim.SGD(models[m].parameters(), lr=1e-3, momentum=0.9)
    ```

##### Results

- English-Deutsch
    - after 300 iterations
        ```
        BLEU = 21.27, 49.5/27.2/17.9/12.4
        ```
    - after 600 iterations
        ```
        BLEU = 21.39, 49.1/26.8/17.6/12.2
        ```
    - after 900 iterations
        ```
        BLEU = 21.49, 49.0/26.8/17.6/12.2
        ```

- Deutsch-English
    - after 300 iterations
        ```
        BLEU = 25.90, 56.3/33.0/22.4/15.9
        ```
    - after 600 iterations
        ```
        BLEU = 25.89, 56.0/32.8/22.3/15.8
        ```
    - after 900 iterations
        ```
        BLEU = 25.91, 56.1/32.9/22.3/15.8
        ```

##### Comparisons

| Model        | Original | iter300 | iter600 | iter900 | iter1200 |
|--------------|---------:|--------:|--------:|--------:|---------:|
| EN-DE        | 20.54    | 21.27   | 21.39   | 21.49   |          |
| DE-EN        | 24.69    | 25.90   | 25.89   | 25.91   |          |
| EN-DE (bleu) |          | 21.42   | 21.57   | 21.55   | 21.55    |
| DE-EN (bleu) |          | 25.96   | 26.25   | 26.22   | 26.18    |
