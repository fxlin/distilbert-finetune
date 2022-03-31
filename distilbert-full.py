#!/usr/bin/env python3
#!pip install datasets transformers[sentencepiece]
'''
cf: https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/chapter3/section4.ipynb#scrollTo=Dp2xjrNUjXNL
sst: Stanford Sentiment Treebank
'''

from datasets import load_dataset, ClassLabel, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import torch
import os

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
metric = load_metric("accuracy")

# xzl: this???
os.environ["WANDB_DISABLED"] = "true"

def to_bin_class(ex):
	ex['label'] = round(ex['label'])
	return ex

def tokenize_fn(ex):
	return tokenizer(ex['sentence'], padding='max_length', truncation=True)

def compute_metrics(eval_pred):
	logits, labels = eval_pred
	preds = np.argmax(logits, axis=-1)
	return metric.compute(predictions=preds, references=labels)


sst = load_dataset('sst', 'default')
sst = sst.remove_columns(['tokens', 'tree'])
sst = sst.map(to_bin_class)
sst = sst.cast_column('label', ClassLabel(num_classes=2))

sst_tokenized = sst.map(tokenize_fn, batched=True)

# xzl: remove more
sst_tokenized = sst_tokenized.remove_columns(['sentence'])

trn_set = sst_tokenized['train']
tst_set = sst_tokenized['test']
val_set = sst_tokenized['validation']

print(val_set.column_names)

# batch ...
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    trn_set, shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    val_set, batch_size=8, collate_fn=data_collator
)

# check data ...
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}    

# run one batch
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

# ... prep optim lr scheduler etc  ... 
#from transformers import AdamW    # deprecated
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(device)

# train

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# validation ...

model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

v=metric.compute()
print(v)