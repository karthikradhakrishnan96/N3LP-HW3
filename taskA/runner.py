import os
import time

import torch
from torch import nn
from torchtext.data import BucketIterator

import sys

sys.path.insert(1, "./")

from constants.constants import *
from sesame_street.tokenizers.BertTokenizer import BertTokenizer
from sesame_street.utils.BertWeightLoader import BertWeightLoader
from taskA.datasets.RumorDataset import RumorDataset
from taskA.wrappers.ModelWrapper import ModelWrapper
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
        y_batch = ele.labels.to(device)
        y_pred = model.forward(x_batch_input_ids, x_batch_token_ids, x_batch_mask_ids)
        loss = criterion(y_pred, y_batch)
        running_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if (batch_idx + 1) % UPDATE_STEPS == 0:
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
    with torch.no_grad():
        for batch_idx, ele in enumerate(iterator):
            x_batch_input_ids = ele.input_ids.to(device)
            x_batch_token_ids = ele.token_ids.to(device)
            x_batch_mask_ids = ele.mask_ids.to(device)
            y_batch = ele.labels.to(device)
            y_pred = model.forward(x_batch_input_ids, x_batch_token_ids, x_batch_mask_ids)
            loss = criterion(y_pred, y_batch)
            running_loss += loss.item()
            _, y_pred = torch.max(y_pred, 1)
            num_correct += (y_pred == y_batch).sum().item()
            num_total += len(y_pred)
            predictions.extend(y_pred.detach().cpu().tolist())
            gold.extend(y_batch.detach().cpu().tolist())
    f1 = f1_score(gold, predictions, average='macro').item()
    print("num_correct %d num_total %d acc %.3f loss %.3f F1 %.6f" % (
        num_correct, num_total, num_correct / num_total, running_loss / num_total, f1))


def get_class_weights(examples, num_classes):
    sums = torch.zeros(num_classes)
    for example in examples:
        label = example.labels
        sums[label] += 1
    return sums.max().expand(num_classes) / sums


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("Preparing datasets")
    train_set = RumorDataset(os.path.sep.join(TRAIN_PATH), tokenizer)
    dev_set = RumorDataset(os.path.sep.join(DEV_PATH), tokenizer)
    train_iter = BucketIterator(train_set, sort_key=lambda example: -len(example.input_ids), sort=True,
                                shuffle=False,
                                batch_size=BATCH_SIZE, repeat=False,
                                device=device)
    dev_iter = BucketIterator(dev_set, sort_key=lambda example: -len(example.input_ids), sort=True,
                              shuffle=False,
                              batch_size=BATCH_SIZE, repeat=False,
                              device=device)
    model = ModelWrapper()
    model = model.to(device)
    # optimizer = BertAdam(model.parameters(), LEARNING_RATE)
    optimizer = torch.optim.AdamW(model.parameters(), LEARNING_RATE)
    weights = torch.load(os.path.sep.join(MODEL_PATH))
    BertWeightLoader.from_hugging_face(model.bert, weights)
    class_weights = get_class_weights(train_set.examples, NUM_CLASSES)
    print("Using class weights : ", class_weights.detach().cpu().numpy().tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    print("Starting to train")
    for epoch in range(EPOCHS):
        train(model, train_iter, criterion, optimizer)
        validate(model, dev_iter, criterion)
