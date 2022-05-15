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

# def drawplot(classifier_step_acc, config_num_NB, skipratio_valacc):
#     plt.figure()
#     plt.title("classifier accuracy vs steps {}".format(config_num_NB))
#     plt.plot(range(len(classifier_step_acc[0])), classifier_step_acc[0], label='Thre=0.1 SkipRatio={:.2f}% ValAcc={:.3f}'.format(skipratio_valacc[0][0], skipratio_valacc[0][1]))
#     # plt.text(len(classifier_step_acc[0]), classifier_step_acc[0][-1], '{:.2f}%_{:.3f}'.format(skipratio_valacc[0][0], skipratio_valacc[0][1]), fontsize=7)
#     plt.plot(range(len(classifier_step_acc[1])), classifier_step_acc[1], label='Thre=0.2 SkipRatio={:.2f}% ValAcc={:.3f}'.format(skipratio_valacc[1][0], skipratio_valacc[1][1]))
#     # plt.text(len(classifier_step_acc[1]), classifier_step_acc[1][-1],'{:.2f}%_{:.3f}'.format(skipratio_valacc[1][0], skipratio_valacc[1][1]), fontsize=7)
#     plt.plot(range(len(classifier_step_acc[2])), classifier_step_acc[2], label='Thre=0.3 SkipRatio={:.2f}% ValAcc={:.3f}'.format(skipratio_valacc[2][0], skipratio_valacc[2][1]))
#     # plt.text(len(classifier_step_acc[2]), classifier_step_acc[2][-1],'{:.2f}%_{:.3f}'.format(skipratio_valacc[2][0], skipratio_valacc[2][1]), fontsize=7)
#     plt.plot(range(len(classifier_step_acc[3])), classifier_step_acc[3], label='Thre=0.4 SkipRatio={:.2f}% ValAcc={:.3f}'.format(skipratio_valacc[3][0], skipratio_valacc[3][1]))
#     # plt.text(len(classifier_step_acc[3]), classifier_step_acc[3][-1],'{:.2f}%_{:.3f}'.format(skipratio_valacc[3][0], skipratio_valacc[3][1]), fontsize=7)
#     plt.plot(range(len(classifier_step_acc[4])), classifier_step_acc[4], label='Thre=0.5 SkipRatio={:.2f}% ValAcc={:.3f}'.format(skipratio_valacc[4][0], skipratio_valacc[4][1]))
#     # plt.text(len(classifier_step_acc[4]), classifier_step_acc[4][-1],'{:.2f}%_{:.3f}'.format(skipratio_valacc[4][0], skipratio_valacc[4][1]), fontsize=7)
#     plt.plot(range(len(classifier_step_acc[5])), classifier_step_acc[5], label='Thre=0.6 SkipRatio={:.2f}% ValAcc={:.3f}'.format(skipratio_valacc[5][0], skipratio_valacc[5][1]))
#     # plt.text(len(classifier_step_acc[5]), classifier_step_acc[5][-1],'{:.2f}%_{:.3f}'.format(skipratio_valacc[5][0], skipratio_valacc[5][1]), fontsize=7)
#     plt.xlabel('test steps (of training set)')
#     plt.ylabel('classifier prediction accuracy')
#     plt.grid()
#     plt.legend()
#     plt.savefig('{}_classifier_train_step_1adp2fix.png'.format(config_num_NB))
#     plt.show()
#
# # skipratio_valacc_clsacc = []
# # for step in [50, 100, 200, 300, 400, 500]:
# #     # for fix_t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
# #     skipratio_valacc_clsacc.append(np.load('1adp2adp/{}_cls_step_SkipRatio_ValAcc_ClsAcc.npy'.format(step)))
# # print(skipratio_valacc_clsacc)
#
# classifier_step_acc_50 = []
# skipratio_valacc_50 = []
# for i in [0.1,0.2,0.3,0.4,0.5,0.6]:
#     classifier_step_acc_50.append(np.load('1adp2fix/50_cls_step_thre_{}_ClsAcc_all.npy'.format(i)))
#     skipratio_valacc_50.append(np.load('1adp2fix/{}_cls_step_thre_{}_SkipRatio_ValAcc_ClsAcc.npy'.format(50, i))[0:2].tolist())
# drawplot(classifier_step_acc_50, 50, skipratio_valacc_50)
#
# classifier_step_acc_100 = []
# skipratio_valacc_100 = []
# for i in [0.1,0.2,0.3,0.4,0.5,0.6]:
#     classifier_step_acc_100.append(np.load('1adp2fix/100_cls_step_thre_{}_ClsAcc_all.npy'.format(i)))
#     skipratio_valacc_100.append(np.load('1adp2fix/{}_cls_step_thre_{}_SkipRatio_ValAcc_ClsAcc.npy'.format(100, i))[0:2].tolist())
# drawplot(classifier_step_acc_100, 100, skipratio_valacc_100)
#
# classifier_step_acc_200 = []
# skipratio_valacc_200 = []
# for i in [0.1,0.2,0.3,0.4,0.5,0.6]:
#     classifier_step_acc_200.append(np.load('1adp2fix/200_cls_step_thre_{}_ClsAcc_all.npy'.format(i)))
#     skipratio_valacc_200.append(np.load('1adp2fix/{}_cls_step_thre_{}_SkipRatio_ValAcc_ClsAcc.npy'.format(200, i))[0:2].tolist())
# drawplot(classifier_step_acc_200, 200, skipratio_valacc_200)
#
# classifier_step_acc_300 = []
# skipratio_valacc_300 = []
# for i in [0.1,0.2,0.3,0.4,0.5,0.6]:
#     classifier_step_acc_300.append(np.load('1adp2fix/300_cls_step_thre_{}_ClsAcc_all.npy'.format(i)))
#     skipratio_valacc_300.append(np.load('1adp2fix/{}_cls_step_thre_{}_SkipRatio_ValAcc_ClsAcc.npy'.format(300, i))[0:2].tolist())
# drawplot(classifier_step_acc_300, 300, skipratio_valacc_300)
#
# classifier_step_acc_400 = []
# skipratio_valacc_400 = []
# for i in [0.1,0.2,0.3,0.4,0.5,0.6]:
#     classifier_step_acc_400.append(np.load('1adp2fix/400_cls_step_thre_{}_ClsAcc_all.npy'.format(i)))
#     skipratio_valacc_400.append(np.load('1adp2fix/{}_cls_step_thre_{}_SkipRatio_ValAcc_ClsAcc.npy'.format(400, i))[0:2].tolist())
# drawplot(classifier_step_acc_400, 400, skipratio_valacc_400)
#
# classifier_step_acc_500 = []
# skipratio_valacc_500 = []
# for i in [0.1,0.2,0.3,0.4,0.5,0.6]:
#     classifier_step_acc_500.append(np.load('1adp2fix/500_cls_step_thre_{}_ClsAcc_all.npy'.format(i)))
#     skipratio_valacc_500.append(np.load('1adp2fix/{}_cls_step_thre_{}_SkipRatio_ValAcc_ClsAcc.npy'.format(500, i))[0:2].tolist())
# drawplot(classifier_step_acc_500, 500, skipratio_valacc_500)

# for i in range(6):
#     print(classifier_step_acc_50[i][-1])
#     print(classifier_step_acc_100[i][-1])
#     print(classifier_step_acc_200[i][-1])
#     print(classifier_step_acc_300[i][-1])
#     print(classifier_step_acc_400[i][-1])
#     print(classifier_step_acc_500[i][-1])

# def drawplot2(classifier_step_acc, config_num_NB, skipratio_valacc):
#     plt.figure()
#     plt.title("classifier accuracy vs steps {}".format(config_num_NB))
#     plt.plot(range(len(classifier_step_acc[0])), classifier_step_acc[0], label='SkipRatio={:.2f}% ValAcc={:.3f}'.format(skipratio_valacc[0][0], skipratio_valacc[0][1]))
#     plt.xlabel('test steps (of training set)')
#     plt.ylabel('classifier prediction accuracy')
#     plt.grid()
#     plt.legend()
#     plt.savefig('{}_classifier_train_step_1adp2fix0.png'.format(config_num_NB))
#     plt.show()
#
# # skipratio_valacc_clsacc = []
# # for step in [50, 100, 200, 300, 400, 500]:
# #     # for fix_t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
# #     skipratio_valacc_clsacc.append(np.load('1adp2adp/{}_cls_step_SkipRatio_ValAcc_ClsAcc.npy'.format(step)))
# # print(skipratio_valacc_clsacc)
#
# classifier_step_acc_50 = []
# skipratio_valacc_50 = []
# # for i in [0.1,0.2,0.3,0.4,0.5,0.6]:
# classifier_step_acc_50.append(np.load('1adp2fix0/50_cls_step_ClsAcc_all.npy'))
# skipratio_valacc_50.append(np.load('1adp2fix0/{}_cls_step_SkipRatio_ValAcc_ClsAcc.npy'.format(50))[0:2].tolist())
# drawplot2(classifier_step_acc_50, 50, skipratio_valacc_50)
#
# classifier_step_acc_100 = []
# skipratio_valacc_100 = []
# # for i in [0.1,0.2,0.3,0.4,0.5,0.6]:
# classifier_step_acc_100.append(np.load('1adp2fix0/100_cls_step_ClsAcc_all.npy'))
# skipratio_valacc_100.append(np.load('1adp2fix0/{}_cls_step_SkipRatio_ValAcc_ClsAcc.npy'.format(100))[0:2].tolist())
# drawplot2(classifier_step_acc_100, 100, skipratio_valacc_100)
#
# classifier_step_acc_200 = []
# skipratio_valacc_200 = []
# # for i in [0.1,0.2,0.3,0.4,0.5,0.6]:
# classifier_step_acc_200.append(np.load('1adp2fix0/200_cls_step_ClsAcc_all.npy'))
# skipratio_valacc_200.append(np.load('1adp2fix0/{}_cls_step_SkipRatio_ValAcc_ClsAcc.npy'.format(200))[0:2].tolist())
# drawplot2(classifier_step_acc_200, 200, skipratio_valacc_200)
#
# classifier_step_acc_300 = []
# skipratio_valacc_300 = []
# # for i in [0.1,0.2,0.3,0.4,0.5,0.6]:
# classifier_step_acc_300.append(np.load('1adp2fix0/300_cls_step_ClsAcc_all.npy'))
# skipratio_valacc_300.append(np.load('1adp2fix0/{}_cls_step_SkipRatio_ValAcc_ClsAcc.npy'.format(300))[0:2].tolist())
# drawplot2(classifier_step_acc_300, 300, skipratio_valacc_300)
#
# classifier_step_acc_400 = []
# skipratio_valacc_400 = []
# # for i in [0.1,0.2,0.3,0.4,0.5,0.6]:
# classifier_step_acc_400.append(np.load('1adp2fix0/400_cls_step_ClsAcc_all.npy'))
# skipratio_valacc_400.append(np.load('1adp2fix0/{}_cls_step_SkipRatio_ValAcc_ClsAcc.npy'.format(400))[0:2].tolist())
# drawplot2(classifier_step_acc_400, 400, skipratio_valacc_400)
#
# classifier_step_acc_500 = []
# skipratio_valacc_500 = []
# # for i in [0.1,0.2,0.3,0.4,0.5,0.6]:
# classifier_step_acc_500.append(np.load('1adp2fix0/500_cls_step_ClsAcc_all.npy'))
# skipratio_valacc_500.append(np.load('1adp2fix0/{}_cls_step_SkipRatio_ValAcc_ClsAcc.npy'.format(500))[0:2].tolist())
# drawplot2(classifier_step_acc_500, 500, skipratio_valacc_500)

def drawplot3(classifier_step_acc, stage1, skipratio_valacc):
    plt.figure()
    plt.title("classifier accuracy vs Stage1 steps {}".format(stage1))
    plt.plot(range(len(classifier_step_acc[0])), classifier_step_acc[0], label='Stage0=300 SkipRatio={:.2f}% ValAcc={:.3f}'.format(skipratio_valacc[0][0], skipratio_valacc[0][1]))
    plt.plot(range(len(classifier_step_acc[1])), classifier_step_acc[1], label='Stage0=400 SkipRatio={:.2f}% ValAcc={:.3f}'.format(skipratio_valacc[1][0], skipratio_valacc[1][1]))
    plt.plot(range(len(classifier_step_acc[2])), classifier_step_acc[2], label='Stage0=500 SkipRatio={:.2f}% ValAcc={:.3f}'.format(skipratio_valacc[2][0], skipratio_valacc[2][1]))
    plt.xlabel('test steps (of training set)')
    plt.ylabel('classifier prediction accuracy')
    plt.grid()
    plt.legend()
    plt.savefig('0find1adp2fix/{}_stage1_0find1adp2fix.png'.format(stage1))
    plt.show()

for i in [160, 180, 220, 240, 260, 280, 320, 340, 360, 380]:
    classifier_step_acc = []
    skipratio_valacc = []
    for j in [300, 400, 500]:
        classifier_step_acc.append(np.load('0find1adp2fix/{}_stage0_{}_stage1_ClsAcc_all.npy'.format(j, i)))
        skipratio_valacc.append(np.load('0find1adp2fix/{}_stage0_{}_stage1_SkipRatio_ValAcc_ClsAcc.npy'.format(j, i))[0:2].tolist())
    drawplot3(classifier_step_acc, i, skipratio_valacc)


# skipratio_valacc_clsacc = []
# for step in [50, 100, 200, 300, 400, 500]:
#     # for fix_t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
#     skipratio_valacc_clsacc.append(np.load('1adp2adp/{}_cls_step_SkipRatio_ValAcc_ClsAcc.npy'.format(step)))
# print(skipratio_valacc_clsacc)

# classifier_step_acc_50 = []
# skipratio_valacc_50 = []
# for i in [50, 100, 200, 300, 400, 500]:
#     classifier_step_acc_50.append(np.load('0find1adp2fix/{:.1f}_stage0_50_stage1_ClsAcc_all.npy'.format(i)))
#     skipratio_valacc_50.append(np.load('0find1adp2fix/{:.1f}_stage0_50_stage1_SkipRatio_ValAcc_ClsAcc.npy'.format(i))[0:2].tolist())
# drawplot3(classifier_step_acc_50, 50, skipratio_valacc_50)
#
# classifier_step_acc_100 = []
# skipratio_valacc_100 = []
# for i in [50, 100, 200, 300, 400, 500]:
#     classifier_step_acc_100.append(np.load('0find1adp2fix/{:.1f}_stage0_100_stage1_ClsAcc_all.npy'.format(i)))
#     skipratio_valacc_100.append(np.load('0find1adp2fix/{:.1f}_stage0_100_stage1_SkipRatio_ValAcc_ClsAcc.npy'.format(i))[0:2].tolist())
# drawplot3(classifier_step_acc_100, 100, skipratio_valacc_100)
#
# classifier_step_acc_200 = []
# skipratio_valacc_200 = []
# for i in [50, 100, 200, 300, 400, 500]:
#     classifier_step_acc_200.append(np.load('0find1adp2fix/{:.1f}_stage0_200_stage1_ClsAcc_all.npy'.format(i)))
#     skipratio_valacc_200.append(np.load('0find1adp2fix/{:.1f}_stage0_200_stage1_SkipRatio_ValAcc_ClsAcc.npy'.format(i))[0:2].tolist())
# drawplot3(classifier_step_acc_200, 200, skipratio_valacc_200)
#
# classifier_step_acc_300 = []
# skipratio_valacc_300 = []
# for i in [50, 100, 200, 300, 400, 500]:
#     classifier_step_acc_300.append(np.load('0find1adp2fix/{:.1f}_stage0_300_stage1_ClsAcc_all.npy'.format(i)))
#     skipratio_valacc_300.append(np.load('0find1adp2fix/{:.1f}_stage0_300_stage1_SkipRatio_ValAcc_ClsAcc.npy'.format(i))[0:2].tolist())
# drawplot3(classifier_step_acc_300, 300, skipratio_valacc_300)
#
# classifier_step_acc_400 = []
# skipratio_valacc_400 = []
# for i in [50, 100, 200, 300, 400, 500]:
#     classifier_step_acc_400.append(np.load('0find1adp2fix/{:.1f}_stage0_400_stage1_ClsAcc_all.npy'.format(i)))
#     skipratio_valacc_400.append(np.load('0find1adp2fix/{:.1f}_stage0_400_stage1_SkipRatio_ValAcc_ClsAcc.npy'.format(i))[0:2].tolist())
# drawplot3(classifier_step_acc_400, 400, skipratio_valacc_400)
#
# classifier_step_acc_500 = []
# skipratio_valacc_500 = []
# for i in [50, 100, 200, 300, 400, 500]:
#     classifier_step_acc_500.append(np.load('0find1adp2fix/{:.1f}_stage0_500_stage1_ClsAcc_all.npy'.format(i)))
#     skipratio_valacc_500.append(np.load('0find1adp2fix/{:.1f}_stage0_500_stage1_SkipRatio_ValAcc_ClsAcc.npy'.format(i))[0:2].tolist())
# drawplot3(classifier_step_acc_500, 500, skipratio_valacc_500)