import numpy as np
import os
import pandas as pd
import re
import torch
from torch.utils.data import (
    DataLoader,
    Dataset
)

from collections import Counter
from tqdm.autonotebook import tqdm
from utils import (
    BASE_DIR,
    CHAR_SPEC,
    CLASSES,
    FREQ_BND,
    MAX_LEN,
    META_PATH,
    PAD_METHOD,
    SEED,
    SPLITS,
    TOKS_SPEC,
    dump_pickle,
    load_pickle
)


class TextDataset(Dataset):
    
    def __init__(self, 
                 meta,
                 processor,
                 tokenizer):
        super(TextDataset, self).__init__()
        self.meta = pd.read_csv(meta) if isinstance(meta, str) else meta
        self.all_cls = self.meta["class"].unique()
        self.num_classes = len(self.all_cls)
        if self.meta["class"].dtype != "int":
            cls_map = dict(zip(self.all_cls, range(len(self.all_cls))))
            self.meta["class"] = self.meta["class"].map(cls_map)
        self.processor = processor
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.meta.iloc[idx, 1]
        textpath = self.meta.iloc[idx, 0]
        text = self.processor.process_text(textpath)
        tokens = self.tokenizer.transform(text)
        tokens = torch.Tensor(tokens)
        return tokens, label



class TextTokenizer:
    
    def __init__(self,
                 freq_bnd=FREQ_BND,
                 pad_method=PAD_METHOD,
                 max_len=MAX_LEN,
                 seed=SEED):
        self.freq_bnd = freq_bnd
        self.pad_method = pad_method
        self.max_len = max_len
        self.tok2id = {tok: tok_id for tok_id, tok in enumerate(TOKS_SPEC)}
        self.id2tok = {tok_id: tok for tok_id, tok in enumerate(TOKS_SPEC[:-1])}
        self.seed = seed
        self.spec_el = dict(PAD_EL = [self.tok2id["pad_tok"]],
                            EOS_EL = [self.tok2id["eos_tok"]],
                            SOS_EL = [self.tok2id["sos_tok"]])
        self.pad_params = ()
        self.wordcount = {}
        
    
    def count_words(self, X):
        all_words = []
        for text in X:
            all_words.extend(text)
        self.wordcount = Counter(all_words)
    
    def get_dicts(self):
        cnt = 4
        for word, freq in self.wordcount.items():
            if freq > self.freq_bnd:
                self.tok2id[word] = cnt
            else:
                self.tok2id[word] = self.tok2id["unk_tok"]
            cnt += 1

        for word, tok in self.tok2id.items():
            if tok == self.tok2id["unk_tok"]:
                self.id2tok[tok] = "unk_tok"
            else:
                self.id2tok[tok] = word
                
    def get_pad_params(self,
                       text_len):
        assert self.pad_method in ["pre", "post", "center", "random"], \
                            "invalid padding method"
        if self.pad_method == "pre":
            self.pad_params = self.max_len - text_len, 0
        elif self.pad_method == "post":
            self.pad_params = 0, self.max_len - text_len
        elif self.pad_method == "center":
            pad_size = (self.max_len - text_len) // 2
            self.pad_params = pad_size, self.max_len - text_len - pad_size
        elif self.pad_method == "random":
            pad_size = np.random.RandomState(self.seed).\
                        randint(0, self.max_len - text_len)
            self.pad_params = pad_size, self.max_len - text_len - pad_size
                 
    def pad_text(self,
                 text):
        text_to_pad = self.spec_el["SOS_EL"] +\
                        text[:self.max_len-2] + self.spec_el["EOS_EL"]
        text_len = len(text_to_pad)
        if text_len == self.max_len:
            return text_to_pad
        else:
            self.get_pad_params(text_len)
            return self.apply_padding(text_to_pad)
        
    def apply_padding(self,
                      text):
        pad_pre, pad_post = self.pad_params
        return self.spec_el["PAD_EL"] * pad_pre + text +\
                    self.spec_el["PAD_EL"] * pad_post
    
    def tokenize_text(self,
                      text):
        tokenized = []
        for word in text:
            if self.tok2id.get(word):
                tokenized.append(self.tok2id[word])
            else:
                tokenized.append(self.tok2id["unk_tok"])
        return self.pad_text(tokenized)
    
    def get_text_from_vec(self,
                          tokens):
        text = []
        for tok in tokens:
            if tok != self.tok2id["sos_tok"] and \
                tok != self.tok2id["eos_tok"] and \
                tok != self.tok2id["pad_tok"]:
                if self.id2tok.get(tok):
                    text.append(self.id2tok[tok])
                else:
                    print(f"{tok} token not found")
                    text.append("unk_tok")
        return text
    
    def fit(self, X):
        self.count_words(X)
        self.get_dicts()
        
    def transform(self, X):
        transformed = []
        if not isinstance(X[0], list):
            X = [X]
        for text in tqdm(X):
            transformed.append(self.tokenize_text(text))
        return transformed
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class TextProcessor:
    
    def __init__(self,
                 corpus,
                 process_fn=None):
        self.meta = 0
        self.corpus = []
        if isinstance(corpus, str) or isinstance(corpus, pd.DataFrame):
            self.meta = corpus
        elif isinstance(corpus, list):
            self.corpus = corpus
            "invalid corpus: pd.DataFrame with metadata, \
                list with texts or path to meta table must be provided"
        self.process_fn = process_fn
        
    def read_file(self, x):
        with open(x, "r") as f:
            return " ".join(f.readlines())

    def clear_text(self, x):
        x = self.read_file(x)
        x = x.replace("<br />", "")
        x = x.lower()
        x = re.sub(CHAR_SPEC, "", x)
        x = x.replace(".", " . ")
        return [tok for tok in x.split() if tok != " "]
    
    def process_text(self, x):
        x = self.clear_text(x)
        if self.process_fn:
            return self.process_fn(x)
        return x

    def create_meta(self,
                    path=None):
        meta_df = []
        for split in SPLITS:
            for cls in CLASSES:
                files_list = os.listdir(os.path.join(BASE_DIR, split, cls))
                meta_df += [(os.path.join(BASE_DIR, split, cls, filepath), cls)
                                for filepath in files_list]
        self.meta = pd.DataFrame(meta_df, columns=["filepath", "class"])
        self.meta["train"] = np.random.RandomState(SEED).binomial(1, 0.8, len(self.meta))
        if not path:
            self.meta.to_csv(META_PATH, index=None)
        else:
            self.meta.to_csv(path, index=None)

    def get_corpus(self):
        if self.corpus:
            return self.corpus
        self.create_meta()
        return [self.process_text(filepath) for filepath in
                tqdm(self.meta["filepath"].values)], self.meta["class"].values.tolist()

            