#chat.py
import torch
import json
from model import CustomGPT
from config import *

# Load tokenizer
with open("tokenizer.json", "r") as f:
    tokens = json.load(f)
stoi = tokens["stoi"]
itos = tokens["itos"]

def encode(text):
    return [stoi[ch] for ch in text if ch in stoi]

def decode(token_ids):
    return "".join(itos[str(i)] for i in token_ids)

# Load model
model = CustomGPT(
    vocab_size,
    n_embd,
    block_size,
    num_layers=4,
    num_heads=num_heads,
    head_size=head_size
).to(device)

ckpt = torch.load("checkpoint.pth", map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

print("Chatbot ready. Type 'exit' to quit.")



while True: 
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    prompt = (
    f"User said: {user_input}\n"
    
)


    idx = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    out = model.generate(
        idx,
        max_new_tokens=200,
        temperature=0.8,
        top_k=50
    )

    full = decode(out[0].tolist())
    response = full.split("Bot:")[-1].strip()

    print("Bot:", response)