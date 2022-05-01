from datasets import load_dataset, ClassLabel, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, get_scheduler, DataCollatorWithPadding
import numpy as np
import torch
import os
# import pandas as pd
import matplotlib.pyplot as plt
# from torchinfo import summary
from torch.utils.data import DataLoader
from torch.optim import AdamW
import time
from tqdm.auto import tqdm

# # load pre-trained
# #  ... many weights are not init'd... random??
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
metric_train = load_metric("accuracy")
metric_test = load_metric("accuracy")

def to_bin_class(ex):
	ex['label'] = round(ex['label'])
	return ex

def tokenize_fn(ex):
	return tokenizer(ex['sentence'], padding='max_length', truncation=True)

def compute_metrics(eval_pred):
	logits, labels = eval_pred
	preds = np.argmax(logits, axis=-1)
	return metric.compute(predictions=preds, references=labels)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class FCGate(torch.nn.Module):
    def __init__(self, in_channels=10, out_channels=8):
        super(FCGate, self).__init__()
        self.linear_layer = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.linear_layer1 = torch.nn.Linear(768, out_channels)
        self.prob_layer = torch.nn.Sigmoid()

    def forward(self,x):
        # print("Before linear", x.shape, x)
        x = self.linear_layer(x)
        # x = self.linear_layer1(x)
        # print("After linear", x.shape, x)
        prob = self.prob_layer(x)
        # print("prob", prob)
        x = (prob > 0.5).float().detach() - \
            prob.detach() + prob
        # x = x.view(x.size(0), 1)
        return x


