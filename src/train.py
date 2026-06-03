# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import matplotlib.pyplot as plt
import torch

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    loss, _ = model(input_batch, targets=target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    num_batches = num_batches or len(data_loader)
    model.eval()
    with torch.no_grad():
        for i, (inp, tgt) in enumerate(data_loader):
            if i >= num_batches:
                break
            loss = calc_loss_batch(inp, tgt, model, device)
            total_loss += loss.item()
    model.train()
    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, global_step, path):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }, path)


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"], ckpt["global_step"]


def generate(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    values, _ = torch.topk(logits, top_k)
                    logits[logits < values[:, -1:]] = float('-inf')
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            if eos_id is not None and next_token.item() == eos_id:
                break
            idx = torch.cat([idx, next_token], dim=1)
    return idx


def generate_and_print_sample(model, tokenizer, device, start_context, max_new_tokens=50, context_size=256, temperature=0.8, top_k=40):
    model.eval()
    encoded = tokenizer.encode(start_context)
    idx = torch.tensor([encoded], dtype=torch.long).to(device)
    out = generate(model, idx, max_new_tokens, context_size, temperature, top_k)
    print(tokenizer.decode(out[0].tolist()))
    model.train()


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer, ckpt_freq=None, start_epoch=0, global_step=0):
    train_losses, val_losses = [], []
    model.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_loss = 0.0
        for inp, tgt in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(inp, tgt, model, device)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            global_step += 1
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        if (epoch + 1) % eval_freq == 0:
            val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
            val_losses.append(val_loss)
            print(f"Epoch {epoch+1} | train loss: {epoch_loss:.4f} | val loss: {val_loss:.4f}")
            generate_and_print_sample(model, tokenizer, device, start_context)

        if ckpt_freq and (epoch + 1) % ckpt_freq == 0:
            save_checkpoint(model, optimizer, epoch, global_step, f"ckpt_epoch{epoch+1}.pt")

    return train_losses


def plot_losses(train_losses, val_losses=None):
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()