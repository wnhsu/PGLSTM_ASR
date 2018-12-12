# A Prioritized Grid LSTM for ASR

This repository contains code to reproduce the results from the paper [A Prioritized Grid Long Short-Term Memory RNN for Speech Recognition](https://groups.csail.mit.edu/sls/publications/2016/WeiNingHsu_SLT_2016.pdf).

To cite this work, please use
```
@inproceedings{hsu2016prioritized,
  title={A prioritized grid long short-term memory RNN for speech recognition},
  author={Hsu, Wei-Ning and Zhang, Yu and Glass, James},
  booktitle={Spoken Language Technology Workshop (SLT), 2016 IEEE},
  pages={467--473},
  year={2016},
  organization={IEEE}
}
```

## Dependencies
This project uses [Kaldi](https://github.com/kaldi-asr/kaldi) for feature extraction, inital HMM-GMM model training, forced alignment, and decoding. Neural network-based acoustic model training was done using [CNTK](https://github.com/Microsoft/CNTK).

## Usage
Place files in Kaldi example script directories (e.g. `kaldi/egs/hkust/s5`) and run:

```sh
cntk_scripts/run_cntk_pglstm_5l.sh \
    --expdir <exp_dir> \
    --ali_src <ali_src> \
    --train_src <train_src> \
    --test_src <test_src> \
    --cn_gpu <cntk_bin> \
```

- exp\_dir: directory to dump experiment results
- ali\_src: directory containing forced alignment results `ali.*.gz`, kaldi GMM model `final.mdl`, and kaldi senone counts file `final.occs`.
- train\_src: directory containing training set features `feats.scp`
- test\_src: directory containing test set features `feats.scp`
- cntk\_bin: path to CNTK binary
