#model.py
import torch
import torch.nn as nn
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.Key = nn.Linear(n_embd, head_size)
        self.Query = nn.Linear(n_embd, head_size)
        self.Value = nn.Linear(n_embd, head_size)
        self.head_size = head_size
        
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask)
        
    def forward(self, x):
        Q = self.Query(x)
        K = self.Key(x)
        V = self.Value(x)
        
        scores = Q @ K.transpose(-2, -1)
        
        scores = scores / (self.head_size ** 0.5)
        
        scores = scores.masked_fill(self.mask[:scores.size(1), :scores.size(2)] == 0, float('-inf'))
        
        weights = torch.softmax(scores, dim=-1)
        
        out = weights @ V
        
        return out
         
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_embd, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        
    def forward(self, x):
        outs = [head(x) for head in self.heads]
        out = torch.cat(outs, dim=-1)
        out = self.proj(out)
        return out
        
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )
    def forward(self, x):
        return self.net(x)
        
class Block(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(num_heads, n_embd, head_size, block_size)
        self.ff = FeedForward(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
        
class CustomGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, num_layers=6, num_heads=4, head_size=32):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        self.block_size = block_size
        
        self.blocks = nn.Sequential(*[
            Block(n_embd, num_heads, head_size, block_size)
            for _ in range(num_layers)
            ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos = torch.arange(0, idx.size(1), device=idx.device)
        pos_emb = self.position_embedding_table(pos)
        
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=100,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0
    ):
        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = idx[:, -self.block_size:]

            # Forward pass
            logits = self(idx_cond)
            logits = logits[:, -1, :]  # last token

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token in set(idx[0].tolist()):
                    logits[0, token] /= repetition_penalty

            # Apply temperature
            logits = logits / temperature

            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)

            # Apply top-k filtering
            if top_k is not None:
                v, ix = torch.topk(probs, top_k)
                mask = torch.zeros_like(probs)
                mask.scatter_(1, ix, 1)
                probs = probs * mask
                probs = probs / probs.sum(dim=-1, keepdim=True)

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative <= top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = True
                filtered = sorted_probs * mask
                filtered = filtered / filtered.sum(dim=-1, keepdim=True)
                probs = torch.zeros_like(probs)
                probs.scatter_(1, sorted_idx, filtered)

            # Sample next token
            next_id = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat((idx, next_id), dim=1)

        return idx

