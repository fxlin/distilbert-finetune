from datasets import load_dataset, ClassLabel, load_metric
from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification, Trainer, TrainingArguments, get_scheduler, DataCollatorWithPadding
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
from dataset import tokenize_fn, to_bin_class, compute_metrics, tokenizer, metric, epoch_time, FCGate
from sklearn.naive_bayes import BernoulliNB
# xzl: this???
os.environ["WANDB_DISABLED"] = "true"

device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
# xzl
# hyperparamters
config_batch_size = 8
config_per_sample = True # otherwise per minibatch
config_n_window = config_batch_size * 16 # mov window avg for cal loss threshold, in num of samples
config_layer_mask = [1,1,1,1,1,1]   # per layer. 1=train, 0=freeze
# config_layer_mask = [0,0,0,0,0,1]   # per layer. 1=train, 0=freeze
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# # load pre-trained
# #  ... many weights are not init'd... random??
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())
print('The DistilBERT model has {:} different named parameters.\n'.format(len(params)))

# embeddeing layers
print('==== Embedding Layer ====\n')
for p in params[0:4]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

# transformer layers
for layer in range(0, 6):
    # 0th layer: params [5..21]. each layer 16 params, 6 layers
    if layer == 0:
        print('\n==== First Transformer ====\n')
    for p in params[4 + 16 * layer: 20 + 16 * layer]:
        if layer == 0:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        if config_layer_mask[layer] == 0:
            p[1].requires_grad = False  # xzl ... freeze
# output layer
print('\n==== Output Layer ====\n')
for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

sst = load_dataset('sst', 'default')
#sst = load_dataset('glue', 'sst2')
sst = sst.remove_columns(['tokens', 'tree'])
sst = sst.map(to_bin_class)
sst = sst.cast_column('label', ClassLabel(num_classes=2))
sst_tokenized = sst.map(tokenize_fn, batched=True)
sst_tokenized = sst_tokenized.remove_columns(['sentence'])

trn_set = sst_tokenized['train']
num_trn = trn_set.num_rows
trn_set_10 = torch.utils.data.Subset(trn_set, list(range(0,int(num_trn/10))))
trn_set_50 = torch.utils.data.Subset(trn_set, list(range(0,int(num_trn/2))))
tst_set = sst_tokenized['test']
val_set = sst_tokenized['validation']
print(val_set.column_names)

# sst2 = load_dataset('glue', 'sst2')
# sst2 = sst2.map(to_bin_class)
# sst2 = sst2.cast_column('label', ClassLabel(num_classes=2))
# sst2_tokenized = sst2.map(tokenize_fn, batched=True)
# sst2_tokenized = sst2_tokenized.remove_columns(['sentence'])

trn_set = sst_tokenized['train']
num_trn = trn_set.num_rows
tst_set = sst_tokenized['test']
val_set = sst_tokenized['validation']

# data loader, split into train/eval sets
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(
   trn_set, shuffle=True, batch_size=config_batch_size, collate_fn=data_collator
   # trn_set_100, shuffle=False, batch_size=config_batch_size, collate_fn=data_collator
    # trn_set_500, shuffle=False, batch_size=config_batch_size, collate_fn=data_collator
    # trn_set_1000, shuffle=False, batch_size=config_batch_size, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    val_set, batch_size=config_batch_size, collate_fn=data_collator
)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 1   # xzl. default:3

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)
model.to(device)

included_batches = list(range(0, len(train_dataloader)))

skip_counter = 0  # how many sample skipped

#loss_history = []
loss_history = np.array([], dtype=np.float32)
loss_history_eff = np.array([], dtype=np.float32)   # effective. actual loss in training

# loss_threshold = 0 # 0.4 # 0.4   # higher -> skip more; lower -> skip less
loss_threshold = 0
loss_threshold_history = []

#for picking examples based on rankings in a minibatch. only backprop these examples
keep_frac = 0.5

# dict()
# list of 1D tensors...
staged_batch = {'input_ids':[], 'attention_mask':[], 'labels':[]}

# train/validation per epoch
loss_values = []
loss_threshold_history.append((0, loss_threshold))
bnb = BernoulliNB(binarize=0.0)
gate = FCGate(in_channels=512*768, out_channels=1).to(device) #num of features
length_loss_list = []
length_list = []

for epoch in range(num_epochs):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, num_epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0
    skip_counter = 0

    progress_bar = tqdm(range(len(train_dataloader) * config_batch_size))

    for step, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        sentence_lengths = []

        # Fwd the whole batch
        model.eval()
        outputs = model(**batch)
        # word_embeddings = model.distilbert.embeddings.word_embeddings(batch['input_ids'])
        # word_embeddings = torch.reshape(word_embeddings, (word_embeddings.shape[0], -1))
        # gate_mask = gate(word_embeddings).squeeze()

        # Naive Bayes Classification
        # for i in range(batch['input_ids'].shape[0]):
        #     print(batch['input_ids'][i].shape)
            # print(batch['input_ids'][i], batch['labels'][i], batch['attention_mask'][i])
            # sentence_lengths.append(torch.count_nonzero(batch['input_ids'][i]).item())

        if config_per_sample:
            loss = torch.nn.CrossEntropyLoss(reduction='none')(outputs.logits,
                                                               batch['labels'])  # per example loss
            # loss history of all samples... discarded or not ...
            loss_history = np.concatenate((loss_history, loss.cpu().detach().numpy()))
            for i in range(len(sentence_lengths)):
                length_loss_list.append(loss[i].item())
                length_list.append(sentence_lengths[i])

            # backprop based on loss threshold
            for idx, l in enumerate(loss):
                if l >= loss_threshold:
                    for k in ['input_ids', 'attention_mask', 'labels']:
                        # staged_batch[k] = torch.cat(staged_batch[k], batch[k][idx])
                        staged_batch[k].append(batch[k][idx])
                else:
                    skip_counter += 1

            # # backprop based on loss threshold
            # for idx, l in enumerate(gate_mask):
            #     if l == 1:
            #         for k in ['input_ids', 'attention_mask', 'labels']:
            #             # staged_batch[k] = torch.cat(staged_batch[k], batch[k][idx])
            #             staged_batch[k].append(batch[k][idx])
            #     else:
            #         skip_counter += 1

                    #######################################
            # adjust loss threshold ... moving avg
            if len(loss_history) > config_n_window:
                loss_threshold = np.average(loss_history[-config_n_window:])
                loss_threshold_history.append((step * config_batch_size - skip_counter, loss_threshold))
            optimizer.zero_grad()  # just in case ...

            # less than 1 batch for backprop ... later
            # n_batches = staged_batch['input_ids'].size(dim=0)
            n_batches = len(staged_batch['input_ids'])
            # print("n_batches = ", n_batches)
            if n_batches < config_batch_size:
                continue

            # has a batch to backprop ... split a batch of 8
            # https://pytorch.org/docs/stable/generated/torch.split.html
            # batch = {'input_ids':[], 'attention_mask':[], 'labels':[]}

            for k in ['input_ids', 'attention_mask', 'labels']:
                batch[k] = torch.stack(staged_batch[k][0:config_batch_size]).to(device)  # already on device??
                staged_batch[k] = staged_batch[k][config_batch_size:]

        else:  # per minibatch
            loss = outputs.loss
            # loss_history.append(loss.cpu().detach().numpy())
            # np.concatenate((loss_history, loss.cpu().detach().numpy()))
            loss_history = np.append(loss_history, loss.item())

            # length_loss_list.append(loss.item())
            # sentence_lengths = sum(sentence_lengths) / len(sentence_lengths)
            # length_list.append(int(sentence_lengths))

            #######################################
            # adjust loss threshold ... moving avg
            win = int(config_n_window / config_batch_size)
            if len(loss_history) > win:
                loss_threshold = np.average(loss_history[-config_n_window:])
                loss_threshold_history.append((step * config_batch_size - skip_counter, loss_threshold))

            if loss.item() < loss_threshold:
                skip_counter += config_batch_size
                optimizer.zero_grad()  # just in case ...
                continue

        model.train()
        outputs = model(**batch)
        loss = outputs.loss

        # loss is a tensor containing a single val
        total_loss += loss.item()
        # loss_history_eff.append((step, loss.item(), 1))   # 1 means backprop
        # loss history of samples for training ... i.e. not discarded ...
        if config_per_sample:
            per_sample_loss = torch.nn.CrossEntropyLoss(reduction='none')(outputs.logits,
                                                                          batch['labels'])  # per example loss
            loss_history_eff = np.concatenate((loss_history_eff, per_sample_loss.cpu().detach().numpy()))
        else:
            loss_history_eff = np.append(loss_history_eff, loss.item())

        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.update(config_batch_size)

    # Calculate the average loss over the training data.
    # avg_train_loss = total_loss / len(train_dataloader)
    avg_train_loss = total_loss / (len(train_dataloader) - skip_counter / config_batch_size)  # xzl

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    print("")
    print(f"  Average training loss: {avg_train_loss:.2f}")
    print("  Training epcoh took: {:}m {:}s".format(*epoch_time(t0, time.time())))
    print(f"  skipped {skip_counter} samples, {100 * skip_counter / config_batch_size / len(train_dataloader):.2f}%",
          "loss_threshold", loss_threshold)

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    print("  Validation took: {:}m {:}s".format(*epoch_time(t0, time.time())))
    print("Accuracy: ", metric.compute())

# plt.title("length vs loss")
# plt.scatter(length_list,length_loss_list)
# plt.show()

# print(loss_history)
plt.hist(loss_history_eff, bins=40)
plt.title("Effective Loss histo")
plt.show()

plt.title("Effective Loss vs # Sample")
plt.plot(loss_history_eff, 'ro', markersize=2)
plt.show()

plt.title("Loss Threshold vs # Sample")
ts = np.transpose(loss_threshold_history)
plt.scatter(ts[0],ts[1])
plt.show()

#print(optimizer.state['grad_history'])

# gradient ... low to high
s = sorted(optimizer.state['grad_history'], key = lambda x: x[1])
# loss  ... low to high
# s = sorted(loss_history, key = lambda x: x[1])

# plt.hist(list(map(lambda x: x[1], optimizer.state['grad_history'])), bins=40)
# plt.title("Gradient histo")
# plt.show()

'''
1. Todos
2. cheap classifier to replace the forward pass at the first place?--> further reduce computation
3. Is it because the pretrain model is so good that even our algorithm heart the accuracy, it still achieves ~84%
4. just update 1 epoch is enough? more epochs?
5. different strategy in third last cells.

double check my understanding
apartment



TODO:
1. check this on some NLP datasets
2. classifier
3. adaptive threshold(later)

Ji
bag of words
'''