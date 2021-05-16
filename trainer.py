import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.autonotebook import tqdm
from utils import (
    BATCH_SIZE,
    LR
)



class NNTrainer:
    
    def __init__(self,
                 data,
                 model,
                 n_epochs,
                 batch_size=BATCH_SIZE,
                 lr=LR,
                 optimizer="Adam",
                 loss_fn="CrossEntropyLoss",
                 device="cpu"):
        self.device = device
        self.train_data = data["train"]
        self.val_data = data["val"]
        self.model = model.to(self.device)
        self.epochs = range(1, n_epochs+1)
        self.loss_fn = getattr(nn, loss_fn)()
        self.optimizer = getattr(optim, optimizer)(self.model.parameters(), lr)
        self.batch_size = batch_size
        self.train_logs = {}
        
    def train(self):
        for epoch in tqdm(self.epochs):
            self.train_epoch(epoch)
            epoch_loss = self.train_logs[epoch]["val_loss"]
            epoch_acc = self.train_logs[epoch]["val_acc"]
            print(f"Epoch {epoch}")
            print(f"Val.loss - {np.round(epoch_loss, 4)}, val.acc. - {np.round(epoch_acc, 4)}")
        return self.model
    
    def train_epoch(self, 
                    epoch):
        
        loss_val = 0.0
        acc_val = 0.0
        sup_val = 0.0
        
        self.model.train()
        
        for x, y in tqdm(self.train_data):
            loss = self.train_batch(x,
                                    y,
                                    train=True)
            
        self.model.eval()
        with torch.no_grad():
            for x, y in tqdm(self.val_data):
                acc, loss = self.train_batch(x,
                                             y)
                loss_val += loss
                acc_val += acc
                sup_val += 1
        
        self.train_logs[epoch] = {}
        self.train_logs[epoch]["val_loss"] = loss_val / sup_val
        self.train_logs[epoch]["val_acc"] = acc_val / sup_val
        
    def train_batch(self,
                    x,
                    y,
                    train=False):
        x = x.to(self.device)
        y = y.to(self.device)
        self.optimizer.zero_grad()
        preds = self.model(x)
        preds = torch.cat([1-preds, preds], dim=1)
        loss = self.loss_fn(preds, y)
        if train:
            loss.backward()
            self.optimizer.step()
            return loss
        else:
            acc = torch.sum(torch.argmax(preds, axis=1) == y) / self.batch_size
            return acc, loss
        
        