import sys
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
from dataset import tokenize_fn, to_bin_class, compute_metrics, tokenizer, metric_train, metric_test, epoch_time, FCGate
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

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# # load pre-trained
# #  ... many weights are not init'd... random??
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
# gate = FCGate(in_channels=512*768, out_channels=1).to(device) #num of features

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

# trn_set = sst_tokenized['train']
# num_trn = trn_set.num_rows
# trn_set_10 = torch.utils.data.Subset(trn_set, list(range(0,int(num_trn/10))))
# trn_set_50 = torch.utils.data.Subset(trn_set, list(range(0,int(num_trn/2))))
# tst_set = sst_tokenized['test']
# val_set = sst_tokenized['validation']
# print(val_set.column_names)

sst2 = load_dataset('glue', 'sst2')
sst2 = sst2.map(to_bin_class)
sst2 = sst2.cast_column('label', ClassLabel(num_classes=2))
sst2_tokenized = sst2.map(tokenize_fn, batched=True)
sst2_tokenized = sst2_tokenized.remove_columns(['sentence'])

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

config_num_NB = int(sys.argv[1])
# fixed_loss_threshold = float(sys.argv[2])
stage0_steps = float(sys.argv[2])
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
length_loss_list = []
length_list = []
classifier_acc = []
classifier_correct = 0
total = 0
stage1_step_counter = 0

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
        print(loss_threshold)
        ###############Stage 0################
        batch = {k: v.to(device) for k, v in batch.items()}
        # Fwd the whole batch
        model.eval()
        outputs = model(**batch)

        loss = torch.nn.CrossEntropyLoss(reduction='none')(outputs.logits,batch['labels'])  # per example loss
        # loss history of all samples... discarded or not ...
        loss_history = np.concatenate((loss_history, loss.cpu().detach().numpy()))
        # if loss_threshold <= fixed_loss_threshold and loss_threshold != -1:

        if step > stage0_steps:
            stage1_step_counter += 1
            # Naive Bayes Classification
            bow = np.zeros((config_batch_size, tokenizer.vocab_size))
            for i in range(batch['input_ids'].shape[0]):
                for index, j in enumerate(batch['input_ids'][i]):
                    if batch['attention_mask'][i][index] == 1:
                        bow[i][j] += 1
            if config_per_sample:
                features = bow
                target = np.array(loss.detach().cpu().numpy() > loss_threshold).astype(np.int32)
                bnb.partial_fit(features,y=target, classes=np.array([0,1]))
                # print(( "stage 1:", loss_threshold))

                ###############Stage 2################
                if stage1_step_counter >= config_num_NB:
                    pred = bnb.predict(features)
                    # print(pred, target)
                    total += config_batch_size
                    classifier_correct += (target == pred).astype(np.int32).sum()
                    print('Classifier Accuracy. Test on train set:', classifier_correct/total)
                    classifier_acc.append(classifier_correct/total)

                    # backprop based on loss threshold
                    for idx, l in enumerate(pred):
                        if l == 1:
                            for k in ['input_ids', 'attention_mask', 'labels']:
                                # staged_batch[k] = torch.cat(staged_batch[k], batch[k][idx])
                                staged_batch[k].append(batch[k][idx])
                        else:
                            skip_counter += 1

                ###############Stage 1################
                else:
                    for idx, l in enumerate(loss):
                        if l >= loss_threshold:
                            for k in ['input_ids', 'attention_mask', 'labels']:
                                # staged_batch[k] = torch.cat(staged_batch[k], batch[k][idx])
                                staged_batch[k].append(batch[k][idx])
                        else:
                            skip_counter += 1
                    # adjust loss threshold ... moving avg
                    if len(loss_history) > config_n_window:
                        loss_threshold = np.average(loss_history[-config_n_window:])
                        loss_threshold_history.append((step * config_batch_size - skip_counter, loss_threshold))

                n_batches = len(staged_batch['input_ids'])
                if n_batches < config_batch_size:
                    continue

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
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        if config_per_sample:
            per_sample_loss = torch.nn.CrossEntropyLoss(reduction='none')(outputs.logits,
                                                                          batch['labels'])  # per example loss
            loss_history_eff = np.concatenate((loss_history_eff, per_sample_loss.cpu().detach().numpy()))
        else:
            loss_history_eff = np.append(loss_history_eff, loss.item())

        if step >= config_num_NB:
            predictions = torch.argmax(outputs.logits, dim=-1)
            metric_train.add_batch(predictions=predictions, references=batch["labels"])
        progress_bar.update(config_batch_size)

    avg_train_loss = total_loss / (len(train_dataloader) - skip_counter / config_batch_size)  # xzl
    loss_values.append(avg_train_loss)

    print("\nTraining Accuracy: ", metric_train.compute())
    print("  Average training loss: {:}".format(avg_train_loss))
    print("  Training epcoh took: {:}m {:}s".format(*epoch_time(t0, time.time())))
    skip_ratio = 100 * skip_counter / config_batch_size / len(train_dataloader)
    print(f"  skipped {skip_counter} samples, {skip_ratio:.2f}%",
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
        metric_test.add_batch(predictions=predictions, references=batch["labels"])

    print("  Validation took: {:}m {:}s".format(*epoch_time(t0, time.time())))
    val_acc = metric_test.compute()['accuracy']
    print("Validation Accuracy: ", val_acc)
    np.save("0find1adp2fix/{}_stage0_{}_stage1_SkipRatio_ValAcc_ClsAcc.npy".format(stage0_steps, config_num_NB), np.array([skip_ratio, val_acc, classifier_acc[-1]]))
    np.save("0find1adp2fix/{}_stage0_{}_stage1_ClsAcc_all.npy".format(stage0_steps, config_num_NB), np.array(classifier_acc))


# print(loss_history)
# plt.hist(loss_history_eff, bins=40)
# plt.title("Effective Loss histo")
# plt.show()
#
# plt.title("Effective Loss vs # Sample")
# plt.plot(loss_history_eff, 'ro', markersize=2)
# plt.show()
#
# plt.title("Loss Threshold vs # Sample")
# ts = np.transpose(loss_threshold_history)
# plt.scatter(ts[0],ts[1])
# plt.show()

#print(optimizer.state['grad_history'])

# gradient ... low to high
# s = sorted(optimizer.state['grad_history'], key = lambda x: x[1])
# loss  ... low to high
# s = sorted(loss_history, key = lambda x: x[1])

# plt.hist(list(map(lambda x: x[1], optimizer.state['grad_history'])), bins=40)
# plt.title("Gradient histo")
# plt.show()