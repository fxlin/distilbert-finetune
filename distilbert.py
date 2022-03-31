#!/usr/bin/env python3
#!pip install datasets

'''
xzl: select which GPUs to use, e.g. 
Trainer API ref: https://huggingface.co/docs/transformers/main_classes/trainer#trainer

CUDA_VISIBLE_DEVICES=1,2,3 python distilbert.py
this avoids run out of GPU memory 


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
trn_set = sst_tokenized['train']
tst_set = sst_tokenized['test']
val_set = sst_tokenized['validation']

#device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

train_args = TrainingArguments('model_dir', evaluation_strategy='epoch')

trainer = Trainer(model=model, args=train_args, train_dataset=trn_set, eval_dataset=val_set, compute_metrics=compute_metrics)

trainer.train()
