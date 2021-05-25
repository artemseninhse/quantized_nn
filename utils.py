import numpy as np
import pandas as pd
import pickle
import random
import torch

from gpt2_classifier import GPT2ForSequenceClassification
from transformers import GPT2Config
from tqdm.autonotebook import tqdm


BASE_DIR = "aclImdb/"
BATCH_SIZE = 256
CHAR_SPEC = '[^a-zA-Z0-9 \n\.]'
CLASSES = ["pos", "neg"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_DIM = 10
FREQ_BND = 50
LR = 1e-4
MAP_LABS = dict(zip([*range(1, 15)], [*range(14)]))
MAX_LEN = 100
N_EPOCHS = 1
PAD_METHOD = "post"
POOL_SIZE = 10
SEED = 1234567890
SPLITS = {
    "train": "data_db/train.csv",
    "test": "data_db/test.csv",
    "quant": "data_db/train.csv",
    "infer": "data_db/infer.csv"
}
STATIC_BATCHES = 100
TEST_SIZE = 0.2
TOKS_SPEC = ["eos_tok", "sos_tok", "pad_tok", "unk_tok"]


def custom_data_gen(data,
                    tokenizer,
                    max_length,
                    batch_size,
                    device):
    cnt = 0
    batch = []
    labs = []
    for lab, text in data:
        batch.append(tokenizer.encode(text.split()[:max_length],
                                      add_special_tokens=True,
                                      padding="max_length",
                                      max_length=max_length,
                                      return_tensors="pt"))
        labs.append(MAP_LABS[lab])
        cnt += 1
        if cnt == batch_size:
            yield torch.cat(batch, axis=0).view(batch_size, max_length).to(device), \
                        torch.Tensor(labs)
            batch = []
            labs = []
            cnt = 0

            
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


def init_data(split):
    meta = pd.read_csv(SPLITS[split])
    data = ((row[0], row[2]) for row in meta.values)
    return data


def init_quant_params(model,
                      data,
                      tokenizer):
    model.eval()
    for num, (x, y) in enumerate(tqdm(custom_data_gen(data,
                                                      tokenizer,
                                                      MAX_LEN,
                                                      BATCH_SIZE,
                                                      DEVICE)),
                                         start=1):
        with torch.no_grad():
            run_batch(x,
                      y,
                      model)
            if num == STATIC_BATCHES:
                break
    return model


def rnd(x):
    return np.round(float(x), 5)


def run_batch(x,
              y,
              model):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        with torch.no_grad():
            preds = model(x).logits
            acc = torch.sum(torch.argmax(preds, dim=-1) == y) / BATCH_SIZE
        return acc
    
    
def calc_accuracy(model,
                  data,
                  tokenizer):
    model.eval()
    batch_val = 0
    acc_val = 0
    for num, (x, y) in enumerate(tqdm(custom_data_gen(data,
                                                      tokenizer,
                                                      MAX_LEN,
                                                      BATCH_SIZE,
                                                      DEVICE)),
                                         start=1):
        acc = run_batch(x,
                        y,
                        model)
        batch_val += 1
        acc_val += acc.item()
    total_acc = acc_val / batch_val
    return total_acc


def measure_inference_time(model,
                           x):
    inf_time = []
    for _ in tqdm(range(100)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        model(x)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        inf_time.append(start.elapsed_time(end))
    return np.mean(inf_time)



