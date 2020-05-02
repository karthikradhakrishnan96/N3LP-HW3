import csv
import os
import time

import torch
from torch import nn
from torchtext.data import BucketIterator

import sys


sys.path.insert(1, "./")

from constants.constants import *
from transformers import BertTokenizer
from taskA.datasets.RumorDataset import RumorDataset
from taskA.wrappers.HfModelWrapper import HfModelWrapper

from sklearn.metrics import f1_score
import numpy as np
import random

print("Setting seeds")

seed = 420 #10601, 11741, 11641, 11747, 42, 69
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

bert_name = BERT_NAME
if SOCIAL:
    bert_name = SOCIAL_BERT_NAME




def train(model, iterator, criterion, optimizer):
    running_loss = 0
    num_correct = 0
    num_total = 0
    start = time.time()
    optimizer.zero_grad()
    model.train()
    for batch_idx, ele in enumerate(iterator):
        updated = False
        x_batch_input_ids = ele.input_ids.to(device)
        x_batch_token_ids = ele.token_ids.to(device)
        x_batch_mask_ids = ele.mask_ids.to(device)
        x_batch_input_ids2 = ele.input_ids2.to(device)
        x_batch_token_ids2 = ele.token_ids2.to(device)
        x_batch_mask_ids2 = ele.mask_ids2.to(device)
        y_batch = ele.labels.to(device)
        y_pred = model.forward(x_batch_input_ids, x_batch_token_ids, x_batch_mask_ids, x_batch_input_ids2, x_batch_token_ids2, x_batch_mask_ids2)
        loss = criterion(y_pred, y_batch) / (BATCH_SIZE * UPDATE_STEPS)
        running_loss += loss.item()
        loss.backward()
        if (batch_idx + 1) % UPDATE_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            updated = True

        _, y_pred = torch.max(y_pred, 1)
        num_correct += (y_pred == y_batch).sum().item()
        num_total += len(y_pred)
        if batch_idx % 50 == 0:
            print("Done with batch %d, acc : %.3f, loss : %.3f" % (
                batch_idx, (num_correct / num_total), running_loss / num_total))
    end = time.time()
    if not updated:
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      optimizer.zero_grad()
    print("Time taken : %d , num_correct %d num_total %d acc %.3f loss %.3f" % (
        end - start, num_correct, num_total, num_correct / num_total, running_loss / num_total))


def validate(model, iterator, criterion):
    running_loss = 0
    num_correct = 0
    num_total = 0
    predictions = []
    gold = []
    model.eval()
    tw_ids = []
    with torch.no_grad():
        for batch_idx, ele in enumerate(iterator):
            x_batch_input_ids = ele.input_ids.to(device)
            x_batch_token_ids = ele.token_ids.to(device)
            x_batch_mask_ids = ele.mask_ids.to(device)
            x_batch_input_ids2 = ele.input_ids2.to(device)
            x_batch_token_ids2 = ele.token_ids2.to(device)
            x_batch_mask_ids2 = ele.mask_ids2.to(device)
            x_batch_tw_ids = ele.tw_ids
            y_batch = ele.labels.to(device)
            y_pred = model.forward(x_batch_input_ids, x_batch_token_ids, x_batch_mask_ids, x_batch_input_ids2, x_batch_token_ids2, x_batch_mask_ids2)
            loss = criterion(y_pred, y_batch)
            running_loss += loss.item()
            _, y_pred = torch.max(y_pred, 1)
            num_correct += (y_pred == y_batch).sum().item()
            num_total += len(y_pred)
            tw_ids.extend(x_batch_tw_ids)
            predictions.extend(y_pred.detach().cpu().tolist())
            gold.extend(y_batch.detach().cpu().tolist())

    f1 = f1_score(gold, predictions, average='macro').item()
    print("num_correct %d num_total %d acc %.3f loss %.3f F1 %.6f" % (
        num_correct, num_total, num_correct / num_total, running_loss / num_total, f1))
    return f1, (x_batch_tw_ids, predictions, gold)


def get_class_weights(examples, num_classes):
    sums = torch.zeros(num_classes)
    for example in examples:
        label = example.labels
        sums[label] += 1
    return sums.max().expand(num_classes) / sums


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    print("Preparing datasets")
    train_set = RumorDataset(os.path.sep.join(TRAIN_PATH), tokenizer)
    dev_set = RumorDataset(os.path.sep.join(DEV_PATH), tokenizer)
    train_iter = BucketIterator(train_set, sort_key=lambda example: -len(example.input_ids), sort=True,
                                shuffle=True,
                                batch_size=BATCH_SIZE, repeat=False,
                                device=device)
    dev_iter = BucketIterator(dev_set, sort_key=lambda example: -len(example.input_ids), sort=True,
                              shuffle=False,
                              batch_size=BATCH_SIZE, repeat=False,
                              device=device)
    model = HfModelWrapper(bert_name)
    model = model.to(device)
    # optimizer = BertAdam(model.parameters(), LEARNING_RATE)
    optimizer = torch.optim.AdamW(model.parameters(), LEARNING_RATE)
    class_weights = get_class_weights(train_set.examples, NUM_CLASSES)
    print("Using class weights : ", class_weights.detach().cpu().numpy().tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction='sum')
    print("Starting to train")
    best_f1 = -1
    best_epoch = -1
    for epoch in range(EPOCHS):
        print("Epoch : ", epoch, " Best F1 : ", best_f1, " @ ", best_epoch)
        train(model, train_iter, criterion, optimizer)
        f1, results = validate(model, dev_iter, criterion)
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            print("Writing...")
            f = open('results-' + str(round(best_f1, ndigits = 5))+".txt", "w")
            writer = csv.writer(f)
            writer.writerows(zip(*results))
            f.close()
