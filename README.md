# SST-2-sentiment-analysis

Use BERT, ALBERT, RoBERTa, XLNet model to classify the SST-2 data set based on pytorch.

These codes are recommended to  run in **Google Colab**.

## Experiment results on test set:

### 1. base model
 Model | Accuracy | Precision	| Recall | F1
 ---- | -----  |----- |----- |----- 
 BERT (base-uncased) | 91.8 |	91.8 |	91.8	| 91.8
RoBERTa (base-uncased)	| **93.4**	| **93.5**	| **93.4**	| **93.3**
ALBERT (base-v2-uncased)	| 91.4	| 91.4	| 91.4	| 91.4
XLNet (base-uncased)	| 92.5	| 92.5	| 92.5	| 92.5

### 2. large model
 Model | Accuracy | Precision	| Recall | F1
 ---- | -----  |----- |----- |----- 
BERT (large-uncased) 	| 93.1	| 93.2	| 93.1	| 93.1
RoBERTa (large-uncased)	| 94.9	| 95.0	| 95.0	| 94.9
ALBERT (large-v2-uncased)	| 92.2	| 92.3	| 92.2	| 92.2
ALBERT (xlarge-v2-uncased)	| 93.8	| 93.8	| 93.9	| 93.8
ALBERT (xxlarge-v2-uncased)	| **95.9**	| **95.9**	| **95.9**	| **95.9**
XLNet (large-uncased)	| 94.6	| 94.7	| 94.6	| 94.6

### 3. base model + text attack
 Model | Accuracy | Precision	| Recall | F1
 ---- | -----  |----- |----- |----- 
 BERT (base-uncased) + textattack |	92.4	|92.8	|92.4	|92.4
RoBERTa (base-uncased) + textattack	|**94.3**	|**94.3**	|**94.3**	|**94.3**
ALBERT (base-uncased) + textattack	|92.0	|92.0|	92.0	|92.0
XLNet (base-uncased) + textattack	|93.7	|93.8	|93.7	|93.7

## LICENSE
Please refer to [MIT License Copyright (c) 2020 YJiangcm](https://github.com/YJiangcm/Movielens1M-Movie-Recommendation-System/blob/main/LICENSE)
