#training.py
import os
import sys
import json
import torch
import random
from config import *
from model import CustomGPT
from datasets import load_dataset

def get_batch(split):
    # 90% Wikitext, 10% sarcasm
    if random.random() < 0.1:
        data = sarcasm_data
    else:
        data = wiki_data

    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,), device=device)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def save_checkpoint(model, optimizer, scaler, iteration, filename="checkpoint.pth"):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "iteration": iteration
    }, filename)
    print(f"Checkpoint saved at iteration {iteration}")
    
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

path = os.path.dirname(__file__)
q_a = os.path.join(path, "data", "q_a_sarcastic.txt")
with open(q_a, "r", encoding="utf-8") as f:
    extra_text = f.read()


# Extract text from train and validation splits
MAX_TRAIN_SAMPLES = 200_000
MAX_VAL_SAMPLES = 10_000
# Prepare two separate datasets
wiki_text = "\n".join(dataset["train"]["text"][:MAX_TRAIN_SAMPLES])
sarcasm_text = extra_text




# Load tokenizer
with open("tokenizer.json", "r") as f:
    tokens = json.load(f)
stoi = tokens["stoi"]
itos = tokens["itos"]

def encode(text):
    return [stoi[ch] for ch in text if ch in stoi]

wiki_data = torch.tensor(encode(wiki_text), dtype=torch.long, device=device)
sarcasm_data = torch.tensor(encode(sarcasm_text), dtype=torch.long, device=device)

model = CustomGPT(
    vocab_size,
    n_embd,
    block_size,
    num_layers=4,
    num_heads=num_heads,
    head_size=head_size
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

scaler = torch.amp.GradScaler("cuda")

if os.path.exists("checkpoint.pth"):
    print("Loading checkpoint...")
    ckpt = torch.load("checkpoint.pth", map_location=device)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    start_iter = ckpt["iteration"]
else:
    print("No checkpoint found, starting fresh.")
    start_iter = 0
def decode(token_ids):
    return "".join(itos[str(i)] for i in token_ids)


    



for iteration in range(start_iter, max_iterations + 1):
    # Switch dataset every 100k iterations
    x, y = get_batch("train")
    x, y = x.to(device), y.to(device)
    
    with torch.amp.autocast("cuda"):
        logits = model(x)
        B, T, V = logits.shape
        loss = torch.nn.functional.cross_entropy(
            logits.view(B*T, V),
            y.view(B*T)
        )

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    if iteration % eval_interval == 0:
        with torch.no_grad():
            x_val, y_val = get_batch("val")
            x_val, y_val = x_val.to(device), y_val.to(device)
            logits_val = model(x_val)
            B, T, V = logits_val.shape
            val_loss = torch.nn.functional.cross_entropy(
                logits_val.view(B*T, V),
                y_val.view(B*T)
            )
        print(f"step {iteration}: train {loss.item():.4f}, val {val_loss.item():.4f}")
        # Auto-save every 25k iterations
    if iteration > 0 and iteration % 5000 == 0:
        save_checkpoint(model, optimizer, scaler, iteration)

