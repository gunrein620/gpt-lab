# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must satisfy 0 <= val_ratio < 1")

    def read_nsmc_tsv(path: str | Path) -> list[dict]:
        rows: list[dict] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.rstrip("\n")
                if line_idx == 0 and line.startswith("id\tdocument\tlabel"):
                    continue
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 3:
                    continue

                document = "\t".join(parts[1:-1]).strip()
                if not document:
                    continue

                try:
                    label = int(parts[-1])
                except ValueError:
                    continue
                if label not in (0, 1):
                    continue

                rows.append({"text": document, "label": label})
        return rows

    train_rows = read_nsmc_tsv(train_tsv_path)
    test_rows = read_nsmc_tsv(test_tsv_path) if test_tsv_path is not None else []

    rng = random.Random(seed)
    rng.shuffle(train_rows)

    val_size = int(len(train_rows) * val_ratio)
    if val_ratio > 0 and len(train_rows) > 1:
        val_size = max(1, val_size)

    val_data = train_rows[:val_size]
    train_data = train_rows[val_size:]
    test_data = test_rows

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        def write_jsonl(path: Path, rows: list[dict]) -> None:
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        write_jsonl(output_path / "nsmc_sentiment_train.jsonl", train_data)
        write_jsonl(output_path / "nsmc_sentiment_val.jsonl", val_data)
        write_jsonl(output_path / "nsmc_sentiment_test.jsonl", test_data)

    return train_data, val_data, test_data


class ReviewSentimentDataset(Dataset):
    """감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다."""

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 128,
        pad_id: int | None = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다."""
        item = self.data[idx]
        try:
            token_ids = self.tokenizer.encode(item["text"], add_bos_eos=True)
        except TypeError:
            token_ids = self.tokenizer.encode(item["text"])

        token_ids = token_ids[: self.max_length]
        if len(token_ids) < self.max_length:
            token_ids = token_ids + [self.pad_id] * (self.max_length - len(token_ids))

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        label = int(item["label"])
        return input_ids, label


class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone 위에 감성 분류용 Linear head를 붙인 모델.

    주의: LM head는 다음 토큰 예측용입니다. 감성 분류는 hidden state 위에 별도 classifier를 붙입니다.
    """

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        self.pad_id = 0
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(gpt_model.config["emb_dim"], num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        x = self.gpt.embedding(input_ids)
        for block in self.gpt.blocks:
            x = block(x, causal_mask=True)
        x = self.gpt.final_norm(x)

        valid_lengths = (input_ids != self.pad_id).sum(dim=1)
        last_token_indices = torch.clamp(valid_lengths - 1, min=0)
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        pooled = x[batch_indices, last_token_indices]

        logits = self.classifier(self.dropout(pooled))
        if labels is None:
            return logits

        labels = labels.to(input_ids.device).long()
        loss = F.cross_entropy(logits, labels)
        return loss, logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for input_batch, label_batch in train_loader:
        input_batch = input_batch.to(device)
        label_batch = label_batch.to(device)

        optimizer.zero_grad()
        loss, logits = model(input_batch, labels=label_batch)
        loss.backward()
        optimizer.step()

        batch_size = input_batch.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == label_batch).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        return float("nan"), float("nan")
    return total_loss / total_examples, total_correct / total_examples


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for input_batch, label_batch in data_loader:
            input_batch = input_batch.to(device)
            label_batch = label_batch.to(device)

            loss, logits = model(input_batch, labels=label_batch)
            batch_size = input_batch.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=-1) == label_batch).sum().item()
            total_examples += batch_size

    if was_training:
        model.train()
    if total_examples == 0:
        return float("nan"), float("nan")
    return total_loss / total_examples, total_correct / total_examples
