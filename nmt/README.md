# Neural Machine Translation

This NMT model is heavily depended on [pcyin/pytorch\_nmt](https://github.com/pcyin/pytorch_nmt). To train model, just follow the steps provided there.

Basiscally, you need to:
1. use `vocab.py` to generate vocab file
2. use `nmt.py` to train model

And you may find `scripts/train.sh` helpful.

### Test Results

##### WMT16

- English-Deutsch
	- with 10% data
		```
		BLEU = 20.54, 49.0/26.7/17.4/11.9 (BP=0.900, ratio=0.904, hyp_len=129552, ref_len=143246)
		```
	- with 100% data
		```
		BLEU = 22.94, 50.9/28.9/19.5/13.8 (BP=0.915, ratio=0.919, hyp_len=131583, ref_len=143246)
		```

- Deutsch-English
	- with 10% data
		```
		BLEU = 24.69, 56.2/32.5/22.0/15.5 (BP=0.880, ratio=0.886, hyp_len=123720, ref_len=139584)
		```
	- with 100% data
		```
		BLEU = 26.73, 57.6/34.4/23.7/17.1 (BP=0.894, ratio=0.899, hyp_len=125477, ref_len=139584)
		```

