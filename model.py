import os
from typing import Tuple, List
from functools import partial

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, BertPreTrainedModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import sqlite3


path = "./"
bert_model_name = 'bert-base-cased'
# path = "../input/jigsaw-toxic-comment-classification-challenge/"
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
assert tokenizer.pad_token_id == 0, "Padding value used in masks is set to zero, please change it everywhere"

class ToxicDatasetSQL(Dataset):
    
    def __init__(self, db_name, table_name, tokenizer: BertTokenizer):
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.pad_token_id

        self.conn = sqlite3.connect(db_name)
        self.table_name = table_name
        self.c = self.conn.cursor()
        self.c.execute("SELECT count(*) FROM " + self.table_name)
        self.len = self.c.fetchall()[0][0]
        self.df = pd.read_sql_query("SELECT my_id, id, comment_text, (toxic | severe_toxic | obscene | threat | insult | identity_hate) AS toxicity  FROM " + self.table_name, self.conn)     
    
    @staticmethod
    def row_to_tensor(tokenizer: BertTokenizer, row: pd.Series) -> Tuple[torch.LongTensor, torch.LongTensor]:
        tokens = tokenizer.encode(row["comment_text"], add_special_tokens=True)
        if len(tokens) > 120:
            tokens = tokens[:119] + [tokens[-1]]
        x = torch.LongTensor(tokens)
        y = torch.FloatTensor(row[["toxicity"]])
        return x, y
        
    
    def __len__(self):
        return self.len

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        return self.row_to_tensor(self.tokenizer, self.df.iloc[index])
            

def collate_fn(batch: List[Tuple[torch.LongTensor, torch.LongTensor]], device: torch.device) \
        -> Tuple[torch.LongTensor, torch.LongTensor]:
    x, y = list(zip(*batch))
    x = pad_sequence(x, batch_first=True, padding_value=0)
    y = torch.stack(y)
    return x.to(device), y.to(device)
    

class BertClassifier(nn.Module):
    
    def __init__(self, bert: BertModel, num_classes: int):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        cls_output = outputs[1] # batch, hidden
        cls_output = self.classifier(cls_output) # batch, 6
        cls_output = torch.sigmoid(cls_output)
        criterion = nn.BCELoss()
        loss = 0
        if labels is not None:
            loss = criterion(cls_output, labels)
        return loss, cls_output
        

def train(model, iterator, optimizer, scheduler):
    model.train()
    total_loss = 0
    for x, y in tqdm(iterator):
        optimizer.zero_grad()
        mask = (x != 0).float()
        loss, outputs = model(x, attention_mask=mask, labels=y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    print("Train loss : {}".format(total_loss / len(iterator)))

def evaluate(model, iterator):
    model.eval()
    pred = []
    true = []
    with torch.no_grad():
        total_loss = 0
        for x, y in tqdm(iterator):
            mask = (x != 0).float()
            loss, outputs = model(x, attention_mask=mask, labels=y)
            total_loss += loss
            true += y.cpu().numpy().tolist()
            pred += outputs.cpu().numpy().tolist()
    true = np.array(true)
    pred = np.array(pred)
    print("ROC AUC : {}".format(roc_auc_score(true, pred)))
    print("Validation loss : {}".format(total_loss / len(iterator)))
 

if __name__ == "__main__":
    dataset = ToxicDatasetSQL("/content/drive/My Drive/Module 3 shared folder/toxic_comment.db", "comment",tokenizer)

    # Creating data indices for training and validation splits:
    validation_split = 0.05
    BATCH_SIZE = 32
    collate_fn = partial(collate_fn, device=device)
    dataset_size = len(dataset)

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(3)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_iterator = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn)
    valid_iterator = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=valid_sampler, collate_fn=collate_fn)

    model = BertClassifier(BertModel.from_pretrained(bert_model_name), 1).to(device)
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    EPOCH_NUM = 2
    # triangular learning rate, linearly grows untill half of first epoch, then linearly decays 
    warmup_steps = 10 ** 3
    total_steps = len(train_iterator) * EPOCH_NUM - warmup_steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)

    for i in range(EPOCH_NUM):
        print('=' * 50, f"EPOCH {i}", '=' * 50)
        train(model, train_iterator, optimizer, scheduler)
        evaluate(model, valid_iterator)
