#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=1,2,3')


# In[2]:


from datasets import load_dataset, ClassLabel, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import torch
import os
from torchinfo import summary

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
metric = load_metric("accuracy")

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

tokenized_datasets = sst.map(tokenize_fn, batched=True)


# In[3]:


from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)


# In[4]:


# test accelerate ...
from accelerate import Accelerator
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

def training_function():
    #global model, optimizer
    
    accelerator = Accelerator()
    checkpoint = "bert-base-uncased"
    
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    optimizer = AdamW(model.parameters(), lr=3e-5)

    train_dl, eval_dl, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dl)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dl:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)           


# In[5]:


from accelerate import notebook_launcher

notebook_launcher(training_function,num_processes=3)


# In[ ]:





# In[ ]:




