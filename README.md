# SST-2-sentiment-analysis

Use BiLSTM_attention, BERT, RoBERTa, XLNet and ALBERT models to classify the SST-2 data set based on pytorch.

These codes are recommended to  run in **Google Colab**, where  you may use free GPU resources.

## 1. Experiment results of BiLSTM_attention models on test set:
The **BiLSTM_attention model** can let us know which words in a sentence do contributions to the sentiment of this sentence. The code is avalibale in "bilstm_attention.ipynb",  where **two types of self-attention mechanism** have been achieved. You can run it in Google Colab for practice. The visualization result is shown below:

<img src="https://github.com/YJiangcm/Movielens1M-Movie-Recommendation-System/blob/main/pictures/attention%E5%8F%AF%E8%A7%86%E5%8C%962.PNG" width="800" height="300">

## 2. Experiment results of BERT models on test set:
For specific BERT models, you can find them from https://huggingface.co/models and then do modify in "models.py".
### 2.1 base model
 Model | Accuracy | Precision	| Recall | F1
 ---- | -----  |----- |----- |----- 
 BERT (base-uncased) | 91.8 |	91.8 |	91.8	| 91.8
RoBERTa (base-uncased)	| **93.4**	| **93.5**	| **93.4**	| **93.3**
XLNet (base-uncased)	| 92.5	| 92.5	| 92.5	| 92.5
ALBERT (base-v2-uncased)	| 91.4	| 91.4	| 91.4	| 91.4

### 2.2 large model
 Model | Accuracy | Precision	| Recall | F1
 ---- | -----  |----- |----- |----- 
BERT (large-uncased) 	| 93.1	| 93.2	| 93.1	| 93.1
RoBERTa (large-uncased)	| 94.9	| 95.0	| 95.0	| 94.9
XLNet (large-uncased)	| 94.6	| 94.7	| 94.6	| 94.6
ALBERT (large-v2-uncased)	| 92.2	| 92.3	| 92.2	| 92.2
ALBERT (xlarge-v2-uncased)	| 93.8	| 93.8	| 93.9	| 93.8
ALBERT (xxlarge-v2-uncased)	| **95.9**	| **95.9**	| **95.9**	| **95.9**

### 2.3 base model + text attack
 Model | Accuracy | Precision	| Recall | F1
 ---- | -----  |----- |----- |----- 
 BERT (base-uncased) + textattack |	92.4	|92.8	|92.4	|92.4
RoBERTa (base-uncased) + textattack	|**94.3**	|**94.3**	|**94.3**	|**94.3**
XLNet (base-uncased) + textattack	|93.7	|93.8	|93.7	|93.7
ALBERT (base-uncased) + textattack	|92.0	|92.0|	92.0	|92.0

## LICENSE
Please refer to [MIT License Copyright (c) 2020 YJiangcm](https://github.com/YJiangcm/Movielens1M-Movie-Recommendation-System/blob/main/LICENSE)
