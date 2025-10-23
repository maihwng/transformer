
import torch
import torch.nn as nn
import torch.nn.functional as F


text = "hello world. this is transformer demo."
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
block_size = 8  
batch_size = 4

def get_batch():
    ix = torch.randint(len(data)-block_size-1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd=32, n_head=2, n_layer=2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

       
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, dim_feedforward=64, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb 
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        out = self.encoder(x, mask)   
        out = self.ln(out)
        logits = self.head(out)      
        return logits

vocab_size = len(chars)
model = MiniTransformer(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(300):
    x, y = get_batch()
    logits = model(x)
    B, T, C = logits.shape
    loss = loss_fn(logits.view(B*T, C), y.view(B*T))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, loss = {loss.item():.4f}")

# ====== THỬ SINH CHUỖI ======
context = torch.tensor([[stoi['h']]], dtype=torch.long)
model.eval()
for _ in range(50):
    logits = model(context[:, -block_size:])
    probs = F.softmax(logits[:, -1, :], dim=-1)
    next_char = torch.multinomial(probs, num_samples=1)
    context = torch.cat((context, next_char), dim=1)
print("Sinh chuỗi:", decode(context[0].tolist()))
