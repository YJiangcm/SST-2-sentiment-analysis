# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:24:49 2020

@author: Jiang Yuxin
"""

from torch.utils.data import Dataset
import torch

class DataPrecessForSentence(Dataset):
    """
    Encoding sentences
    """
    def __init__(self, bert_tokenizer, df, max_word_len = 50):
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = max_word_len
        self.seqs, self.seq_masks, self.seq_segments, self.labels = self.get_input(df)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.seq_masks[idx], self.seq_segments[idx], self.labels[idx]
        
    # Get text and label
    def get_input(self, df):
        """
        Through the process of word segmentation, IDization, truncation, and filling of the input text, 
        the final sequence that can be used for model input is obtained.
        
        Input parameters:
            dataset          : pandas dataframe.
        Output parameters:
            seq              : The'CLS' and'SEP' symbols are respectively spliced at the head and tail of the input parameter seq.
                               If the length is still less than max_seq_len, 0 is used to fill the tail.
            seq_mask         : Only the sequence containing 0, 1 and the length equal to seq is used to characterize whether the symbol in seq is                                  meaningful.
            seq_segment      : Because it is a single sentence, the value is all 0.
            labels           : {0,1}.
        """
        sentences_1 = df['s1'].values
        labels = df['similarity'].values
        # tokenizer
        tokens_seq_1 = list(map(self.bert_tokenizer.tokenize, sentences_1)) # list of shape [sentence_len, token_len]
        # Get fixed-length sequence and its mask
        result = list(map(self.trunate_and_pad, tokens_seq_1))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return torch.Tensor(seqs).type(torch.long), torch.Tensor(seq_masks).type(torch.long),torch.Tensor(seq_segments).type(torch.long), torch.Tensor(labels).type(torch.long)
    
    
    def trunate_and_pad(self, tokens_seq_1):
        # Truncate sequences exceeding the specified length
        if len(tokens_seq_1) > (self.max_seq_len - 2):
            tokens_seq_1 = tokens_seq_1[0:(self.max_seq_len - 2)]
        # Splicing special symbols at the beginning and end respectively
        seq = ['[CLS]'] + tokens_seq_1 + ['[SEP]']
        seq_segment = [0] * (len(tokens_seq_1) + 2)
        # ID
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        # Generate padding sequence according to the length of max_seq_len and seq
        padding = [0] * (self.max_seq_len - len(seq))
        # create seq_mask
        seq_mask = [1] * len(seq) + padding
        # create seq_segment
        seq_segment = seq_segment + padding
        # Stitching filler sequences to seq
        seq += padding
        assert len(seq) == self.max_seq_len
        assert len(seq_mask) == self.max_seq_len
        assert len(seq_segment) == self.max_seq_len
        return seq, seq_mask, seq_segment