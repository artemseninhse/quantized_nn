import numpy as np
import pickle
import random
import torch


BASE_DIR = "aclImdb/"
BATCH_SIZE = 16
CHAR_SPEC = '[^a-zA-Z0-9 \n\.]'
CLASSES = ["pos", "neg"]
EMB_DIM = 10
FREQ_BND = 50
LR = 1e-4
MAX_LEN = 500
META_PATH = "imdb_meta_df.csv"
PAD_METHOD = "post"
POOL_SIZE = 10
SEED = 1234567890
SPLITS = ["train", "test"]
TEST_SIZE = 0.2
TOKS_SPEC = ["eos_tok", "sos_tok", "pad_tok", "unk_tok"]


def dump_pickle(x, path):
    with open(path, "wb") as f:
        pickle.dump(x, f)
        

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

