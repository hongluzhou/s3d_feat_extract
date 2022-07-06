import os
import numpy as np
import pandas as pd
import random
import ffmpeg
import time
import re
import math
import json
import pdb

from nltk.tokenize import RegexpTokenizer
from utils.stop_words import ENGLISH_STOP_WORDS

import torch as th
from torch.utils.data import Dataset


class WikiHow_Step_DataLoader(Dataset):
    """WikiHow Step Data loader."""

    def __init__(
            self,
            args,
            wikihow_subset=1,
            max_words=20,
            token_to_word_path='data/dict.npy'
    ):
        """
        Args:
        """
        wikihow_subset_root = '/export/home/code/ginst/wikihow/WikiHow-DistantSupervision'
        wikihow_full_root = '/export/home/code/ginst/wikihow/WikiHow-Complete/WikiHow-Dataset'
        
        self.max_words = max_words
        self.token_to_word = np.load(os.path.join(os.path.dirname(__file__), token_to_word_path))
        self.word_to_token = {}
        for i, t in enumerate(self.token_to_word):
            self.word_to_token[t] = i + 1  # plus 1 because 0 is used as padding
        
        if wikihow_subset:
            with open(os.path.join(wikihow_subset_root, 'step_label_text.json'), 'r') as f:
                wikihow = json.load(f)
                
            self.step_sentences = [wikihow[article_id][step_id]['headline']
                         for article_id in range(len(wikihow)) 
                         for step_id in range(len(wikihow[article_id]))]
        else:
            print('Not implemented for wikihow full set yet!')
            os._exit(0)
            
        
        print('{} samples...'.format(self.__len__()))  
        
        
        # pdb.set_trace()
        # self.__getitem__(0)
        # pdb.set_trace()
        

    def __len__(self):
        return len(self.step_sentences)

    def text_preprocessing(self, asr, tokenizer=RegexpTokenizer(r"[\w']+")):
        asr = asr.lower()
        asr = ' '.join([word for word in asr.split() if word not in ENGLISH_STOP_WORDS])
        asr = ' '.join([word for word in tokenizer.tokenize(asr) if word not in ENGLISH_STOP_WORDS])
        return asr
    
    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we
        else:
            return th.zeros(self.max_words, dtype=th.long)

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def words_to_ids(self, x):
        return self._words_to_token(self._split_text(x))
    

    def __getitem__(self, idx):
        step_sentence = self.step_sentences[idx]
        step_sentence_preprocessed = self.text_preprocessing(step_sentence)
        step_tokens = self.words_to_ids(step_sentence_preprocessed)
        
        return {'step_id': idx, 
                'step_sentence_preprocessed': step_sentence_preprocessed,
                'step_tokens': step_tokens
               }

    