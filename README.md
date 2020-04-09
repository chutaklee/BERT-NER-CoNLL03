# BERT-NER-CoNLL03
yet another implementation of finetuning BERT on NER task

### Usage
```
python train.py --batch_size=16 --lr=3e-5
```

### Result
```
accuracy:  95.30%; (non-O)
accuracy:  99.15%; precision:  93.70%; recall:  94.80%; FB1:  94.24
              LOC: precision:  96.36%; recall:  96.46%; FB1:  96.41  1839
             MISC: precision:  89.51%; recall:  89.80%; FB1:  89.66  925
              ORG: precision:  90.80%; recall:  91.95%; FB1:  91.37  1358
              PER: precision:  95.24%; recall:  97.72%; FB1:  96.46  1890
micro average precision: 93.69594145043247, recall: 94.7997307303938, f1: 94.24460431654677
```
