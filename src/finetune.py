# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def make_sentiment_dataset(train_tsv_path, test_tsv_path=None, val_ratio=0.08, seed=42, output_dir=None):
    import random
    random.seed(seed)

    def read_tsv(path):
        data = []
        with open(path, encoding="utf-8") as f:
            next(f)  # header skip
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3 and parts[1].strip():
                    data.append({"text": parts[1], "label": int(parts[2])})
        return data

    train_all = read_tsv(train_tsv_path)
    random.shuffle(train_all)
    val_size = int(len(train_all) * val_ratio)
    val_data = train_all[:val_size]
    train_data = train_all[val_size:]
    test_data = read_tsv(test_tsv_path) if test_tsv_path else []

    return train_data, val_data, test_data


class ReviewSentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128, pad_id=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.tokenizer.encode(item["text"])[:self.max_length]
        pad_len = self.max_length - len(encoded)
        encoded = encoded + [self.pad_id] * pad_len
        return torch.tensor(encoded, dtype=torch.long), item["label"]


class GPTForSequenceClassification(nn.Module):
    def __init__(self, gpt_model, num_labels=2, drop_rate=0.1):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(gpt_model.config["emb_dim"], num_labels)

    def forward(self, input_ids, labels=None):
        # GPT embedding + transformer blocks
        x = self.gpt.embedding(input_ids)
        for block in self.gpt.blocks:
            x = block(x)
        x = self.gpt.norm(x)

        # padding이 아닌 마지막 토큰의 hidden state 사용
        pad_id = 0
        mask = (input_ids != pad_id).long()
        last_pos = mask.sum(dim=1) - 1
        batch_size = x.size(0)
        hidden = x[torch.arange(batch_size), last_pos]

        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)

        if labels is None:
            return logits
        loss = F.cross_entropy(logits, labels)
        return loss, logits


def train_epoch_sentiment(model, train_loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for input_ids, labels in train_loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        loss, logits = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(train_loader), correct / total


def evaluate_sentiment(model, data_loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            loss, logits = model(input_ids, labels=labels)
            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(data_loader), correct / total