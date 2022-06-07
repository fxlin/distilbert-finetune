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
import csv
from tqdm.auto import tqdm
from dataset import to_bin_class, compute_metrics, tokenizer, metric_train, metric_test, epoch_time, FCGate
from dataset import tokenize_fn, tokenize_fn_1, tokenize_fn_2, tokenize_fn_3, tokenize_fn_4
from sklearn.naive_bayes import BernoulliNB
# xzl: this???
os.environ["WANDB_DISABLED"] = "true"
# xzl
# hyperparamters
config_batch_size = 8
config_per_sample = True # otherwise per minibatch
config_n_window = config_batch_size * 16 # mov window avg for cal loss threshold, in num of samples
config_layer_mask = [1,1,1,1,1,1]   # per layer. 1=train, 0=freeze
# config_layer_mask = [0,0,0,0,0,1]   # per layer. 1=train, 0=freeze
# config_cls_accuracy = 0.7
config_stage2_start = False
config_cls_window_size = 8

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# # load pre-trained
# #  ... many weights are not init'd... random??
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
# gate = FCGate(in_channels=512*768, out_channels=1).to(device) #num of features

stage0_steps = int(sys.argv[1])
config_cls_loss = float(sys.argv[2])
config_cls_window_size = int(sys.argv[3])
actual_task = str((sys.argv[4]))
gpu_id = str(sys.argv[5])
device = torch.device("cuda:" + gpu_id) if torch.cuda.is_available() else torch.device("cpu")
print(f"This is for task:{actual_task}, steps:{stage0_steps}, 1to2 threshold:{config_cls_loss}, 1to2 windowsize:{config_cls_window_size}")

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

sst = load_dataset('glue', actual_task)
#sst = load_dataset('glue', 'sst2')
if actual_task in ['cola', 'sst2', 'qqp', "mnli_matched", "qnli", "mrpc","stsb","rte"]:
  sst = sst.remove_columns(['idx'])
else:
  sst = sst.remove_columns(['tokens', 'tree'])
sst = sst.map(to_bin_class)
sst = sst.cast_column('label', ClassLabel(num_classes=2))
if  actual_task in ["mrpc","stsb","rte"]:
  sst_tokenized = sst.map(tokenize_fn_1, batched=True)
  sst_tokenized = sst_tokenized.remove_columns(['sentence1','sentence2'])
  
elif actual_task in ['sst', 'sst2', 'cola']:
  sst_tokenized = sst.map(tokenize_fn, batched=True)
  sst_tokenized = sst_tokenized.remove_columns(['sentence'])

elif actual_task in ['qqp']:
  sst_tokenized = sst.map(tokenize_fn_2, batched=True)
  sst_tokenized = sst_tokenized.remove_columns(['question1','question2'])

elif actual_task in ['mnli_matched']:
  sst_tokenized = sst.map(tokenize_fn_3, batched=True)
  sst_tokenized = sst_tokenized.remove_columns(['premise','hypothesis'])

elif actual_task in ['qnli']:
  sst_tokenized = sst.map(tokenize_fn_4, batched=True)
  sst_tokenized = sst_tokenized.remove_columns(['question','sentence'])
  
metric = load_metric("glue", actual_task)

if actual_task in ['mnli_matched']:
    trn_set = sst_tokenized['test']
else:
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

# set stage0 steps from int to percentage
stage0_steps = 0.01 * stage0_steps * num_training_steps
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
classifier_proba = []
classifier_correct = 0
total = 0
step_counter = 0
fixed_adp_threshold = 0

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
        print("Loss threshold: "+str(loss_threshold))
        ###############Stage 0################
        batch = {k: v.to(device) for k, v in batch.items()}
        # Fwd the whole batch
        model.eval()
        outputs = model(**batch)

        loss = torch.nn.CrossEntropyLoss(reduction='none')(outputs.logits,batch['labels'])  # per example loss
        loss_history = np.concatenate((loss_history, loss.cpu().detach().numpy()))
        # adjust loss threshold ... moving avg

        if step_counter <= stage0_steps:
            for idx, l in enumerate(loss):
                if l >= loss_threshold:
                    for k in ['input_ids', 'attention_mask', 'labels']:
                        # staged_batch[k] = torch.cat(staged_batch[k], batch[k][idx])
                        staged_batch[k].append(batch[k][idx])
                else:
                    skip_counter += 1

            if len(loss_history) > config_n_window:
                loss_threshold = np.average(loss_history[-config_n_window:])

            n_batches = len(staged_batch['input_ids'])
            if n_batches < config_batch_size:
                continue

            for k in ['input_ids', 'attention_mask', 'labels']:
                batch[k] = torch.stack(staged_batch[k][0:config_batch_size]).to(device)  # already on device??
                staged_batch[k] = staged_batch[k][config_batch_size:]

        else:
            # Naive Bayes Classification
            bow = np.zeros((batch['input_ids'].shape[0], tokenizer.vocab_size))
            for i in range(batch['input_ids'].shape[0]):
                for index, j in enumerate(batch['input_ids'][i]):
                    if batch['attention_mask'][i][index] == 1:
                        bow[i][j] += 1
            if config_per_sample:
                features = bow
                target = np.array(loss.detach().cpu().numpy() > loss_threshold).astype(np.int32)
                # print(batch['input_ids'].shape[0], len(target))
                bnb.partial_fit(features,y=target, classes=np.array([0,1]))
                pred = bnb.predict(features)
                proba = -np.average(bnb.predict_log_proba(features)[np.arange(batch['input_ids'].shape[0]),target])
                total += config_batch_size
                classifier_correct += (target == pred).astype(np.int32).sum()
                print('Classifier Loss. Test on train set:', proba)
                classifier_acc.append(classifier_correct / total)
                classifier_proba.append(proba)

                ###############Stage 2################
                # if stage1_step_counter >= config_num_NB:
                if config_stage2_start == False and len(classifier_proba) >= config_cls_window_size:
                    print(('Average Classifier Loss in Window', sum(classifier_proba[-config_cls_window_size:]) / config_cls_window_size))
                    if sum(classifier_proba[-config_cls_window_size:]) / config_cls_window_size <= config_cls_loss:
                        config_stage2_start = True
                        print("Switch to Stage2")
                if config_stage2_start:
                    # backprop based on loss threshold
                    # print('stage 22222')
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
                        # print('stage 11111')
                        if l >= loss_threshold:
                            for k in ['input_ids', 'attention_mask', 'labels']:
                                # staged_batch[k] = torch.cat(staged_batch[k], batch[k][idx])
                                staged_batch[k].append(batch[k][idx])
                        else:
                            skip_counter += 1
                    # # adjust loss threshold ... moving avg
                    # if len(loss_history) > config_n_window:
                    #     loss_threshold = np.average(loss_history[-config_n_window:])
                    #     loss_threshold_history.append((step * config_batch_size - skip_counter, loss_threshold))

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

        # if step >= config_num_NB:
        #     predictions = torch.argmax(outputs.logits, dim=-1)
        #     metric_train.add_batch(predictions=predictions, references=batch["labels"])
        step_counter += 1
        progress_bar.update(config_batch_size)

    avg_train_loss = total_loss / (len(train_dataloader) - skip_counter / config_batch_size)  # xzl
    loss_values.append(avg_train_loss)

    # print("\nTraining Accuracy: ", metric_train.compute())
    print("  Average training loss: {:}".format(avg_train_loss))
    print("  Training epcoh took: {:}m {:}s".format(*epoch_time(t0, time.time())))
    skip_ratio = 100 * skip_counter / config_batch_size / len(train_dataloader)
    print(f" skipped {skip_counter} samples, {skip_ratio:.2f}%",
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
    with open('adp0_{}_claloss_Skipratio_Valacc.csv'.format(actual_task), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([stage0_steps, config_cls_loss, config_cls_window_size, config_stage2_start, val_acc, skip_ratio])


# print(loss_history)
# plt.hist(loss_history_eff, bins=40)
# plt.title("Effective Loss histo")
# plt.savefig("plots/0find1adp2fix_fixed/{}_stage0_{}_stage1_EffectiveLossHistogram_{}_{}.png".format(stage0_steps, config_num_NB, actual_task, o_loss_threshold))
# #
# plt.title("Effective Loss vs # Sample")
# plt.plot(loss_history_eff, 'ro', markersize=2)
# plt.savefig("plots/0find1adp2fix_fixed/{}_stage0_{}_stage1_EffectiveLossVSSample_{}_{}.png".format(stage0_steps, config_num_NB, actual_task, o_loss_threshold))
# #
# plt.title("Loss Threshold vs # Sample")
# ts = np.transpose(loss_threshold_history)
# plt.scatter(ts[0],ts[1])
# plt.savefig("plots/0find1adp2fix_fixed/{}_stage0_{}_stage1_LossThresholdVS#Sample_{}_{}.png".format(stage0_steps, config_num_NB, actual_task, o_loss_threshold))
#
# print(optimizer.state['grad_history'])


