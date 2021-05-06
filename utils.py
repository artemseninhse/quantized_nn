import pickle

BASE_DIR = "aclImdb/"
CHAR_SPEC = '[^a-zA-Z0-9 \n\.]'
CLASSES = ["pos", "neg"]
FREQ_BND = 50
MAX_LEN = 500
META_PATH = "imdb_meta_df.csv"
PAD_METHOD = "post"
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

