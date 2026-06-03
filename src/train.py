# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    """TODO: 한 배치를 device로 옮긴 뒤 다음 토큰 예측 cross entropy loss를 계산합니다."""
    # 한 배치를 장치로 옮기고 GPTModel.forward가 반환하는 next-token loss를 사용합니다.
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    loss, _ = model(input_batch, targets=target_batch)
    return loss


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """TODO: data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""
    model.eval()
    total_loss = 0.0
    count = 0
    if num_batches is None:
        num_batches = len(data_loader)
    # 평가 중에는 gradient가 필요 없으므로 no_grad로 메모리와 계산을 줄입니다.
    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
            count += 1
    model.train()
    return total_loss / count if count else float("nan")


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """TODO: model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    # 재시작 가능한 학습을 위해 모델, optimizer, epoch, step을 함께 저장합니다.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """TODO: torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    # 저장된 상태를 같은 장치 기준으로 읽고 모델/optimizer에 다시 주입합니다.
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("global_step", 0))


def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """TODO: temperature와 top-k 샘플링을 지원하는 생성 함수를 구현합니다."""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 긴 문맥은 context_size만큼 잘라 모델의 최대 길이를 넘지 않게 합니다.
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)[:, -1, :]
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                # top-k 밖의 후보를 제거해 낮은 확률 토큰이 샘플링되는 것을 제한합니다.
                top_values, _ = torch.topk(logits, top_k)
                cutoff = top_values[:, [-1]]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))

            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # temperature로 분포의 날카로움을 조절한 뒤 확률적으로 다음 토큰을 뽑습니다.
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_id), dim=1)
            if eos_id is not None and torch.all(next_id == eos_id):
                break
    return idx


def generate_and_print_sample(
    model: GPTModel,
    tokenizer,
    device: torch.device,
    start_context: str,
    max_new_tokens: int = 50,
    context_size: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> None:
    """TODO: start_context를 encode하고 generate 후 decode하여 출력합니다."""
    model.eval()
    # 텍스트 prompt를 token id로 바꾼 뒤 생성 결과를 다시 문자열로 복원합니다.
    ids = tokenizer.encode(start_context, add_bos_eos=False)
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out = generate(model, idx, max_new_tokens, context_size, temperature, top_k)
    print(tokenizer.decode(out[0].tolist(), skip_special=True))


def train_model(
    model: GPTModel,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,
    ckpt_freq: int | None = None,
    start_epoch: int = 0,
    global_step: int = 0,
) -> list[float]:
    """TODO: 사전 학습 루프를 구현하고 epoch별 train loss 리스트를 반환합니다."""
    model.to(device)
    train_losses = []
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for input_batch, target_batch in train_loader:
            # forward -> loss -> backward -> optimizer.step 순서가 기본 학습 루프입니다.
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1
            global_step += 1

            if eval_freq and global_step % eval_freq == 0 and val_loader is not None:
                calc_loss_loader(val_loader, model, device, eval_iter)
                generate_and_print_sample(model, tokenizer, device, start_context)

        train_losses.append(running_loss / batch_count if batch_count else float("nan"))
        if ckpt_freq and (epoch + 1) % ckpt_freq == 0:
            save_checkpoint(model, optimizer, epoch + 1, global_step, f"checkpoint_epoch_{epoch + 1}.pt")
    return train_losses


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """훈련/검증 손실 그래프를 그리는 제공 함수."""
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
