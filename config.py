#config.py
import json
import os
import sys
import torch
path = os.path.join(sys.path[0], "tokenizer.json")
with open(path, "r") as file:
    tokens = json.load(file)
vocab_size = len(tokens["stoi"])
train = 0.9 #training = 90%
validate = 0.1 #validating = 10%
block_size = 128
batch_size = 128
n_embd = 128 #embedding dimension
lr = 3e-4 #learning rate
max_iterations = 15000
eval_interval = 500 #every 1000 iterations it will print the loss
device = "cuda" if torch.cuda.is_available() else "cpu"
num_heads = 4
head_size = n_embd // num_heads
