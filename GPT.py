import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange

with open('C:\\Users\\Menes\\Desktop\\Python\\Andrej_Karpathy\\tinyshakespeare.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for i, char in enumerate(chars)}
encode = lambda w: [stoi[char] for char in w]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype = torch.int64)
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

### hyperparameters ###
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.manual_seed(1337)
torch.manual_seed(1337)
VOCAB_SIZE = len(chars)
BLOCK_SIZE = 128
MAX_ITERS = 1000
EVAL_INTERVAL = 500
LR = 3e-4
NB_HEADS = 6
EVAL_ITERS = 200
BATCH_SIZE = 64
NB_EMB_DIM = 192
DROPOUT = 0.2
###____________________

### helper functions ###
def get_batch(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(low = 0, high = data.shape[0] - BLOCK_SIZE, size = (BATCH_SIZE, ))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in idx])
    y = torch.stack([data[i+1 : i + 1 + BLOCK_SIZE] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.inference_mode()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            loss = model.loss(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

###____________________

### Model ###
class Head(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(in_features = NB_EMB_DIM, out_features = head_size, bias = False)
        self.query = nn.Linear(in_features = NB_EMB_DIM, out_features = head_size, bias = False)
        self.value = nn.Linear(in_features = NB_EMB_DIM, out_features = head_size, bias = False)

        self.dropout = nn.Dropout(p = DROPOUT)

        self.register_buffer('tril', torch.tril((torch.ones(BLOCK_SIZE, BLOCK_SIZE))))
    
    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)   # (B, T, C = HEAD_SIZE)
        q = self.query(x) # (B, T, C = HEAD_SIZE)
        wei = (q @ torch.transpose(k, dim0 = 1, dim1 = 2)) * (C ** -0.5) # (B, T, T)

        # decoder
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)

        wei = F.softmax(wei, dim = 2) # (B, T, T)
        wei = self.dropout(wei)

        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) = (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, nb_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(nb_heads)])
        self.proj = nn.Linear(NB_EMB_DIM, NB_EMB_DIM)
        self.dropout = nn.Dropout(p = DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = 2)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NB_EMB_DIM, 4 * NB_EMB_DIM),
            nn.ReLU(),
            nn.Linear(4 * NB_EMB_DIM, NB_EMB_DIM),
            nn.Dropout(p = DROPOUT)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, nb_heads):
        super().__init__()
        head_size = NB_EMB_DIM // nb_heads
        self.self_att = MultiHeadAttention(head_size = head_size, nb_heads = nb_heads)
        self.feed_forward = FeedForward()
        self.layer_norm1 = nn.LayerNorm(normalized_shape = NB_EMB_DIM)
        self.layer_norm2 = nn.LayerNorm(normalized_shape = NB_EMB_DIM)
    
    def forward(self, x):
        out = x + self.self_att(self.layer_norm1(x))
        return out + self.feed_forward(self.layer_norm2(out))

class GPT(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, NB_EMB_DIM)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, NB_EMB_DIM)

        self.self_att_blocks = nn.Sequential(
            Block(nb_heads = NB_HEADS),
            Block(nb_heads = NB_HEADS),
            Block(nb_heads = NB_HEADS),
            Block(nb_heads = NB_HEADS),
            Block(nb_heads = NB_HEADS),
            Block(nb_heads = NB_HEADS),
            nn.LayerNorm(normalized_shape = NB_EMB_DIM)
        )
    
        self.lang_model_head = nn.Linear(in_features = NB_EMB_DIM, out_features = VOCAB_SIZE)
    
    def forward(self, x):
        B, T = x.shape

        token_emb = self.token_embedding_table(x) # (B, T, C = NB_EMB_DIM)
        position_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T, C = NB_EMB_DIM)
        sum = token_emb + position_emb #(B, T, C = NB_EMB_DIM) by broadcasting
        
        pre_logits = self.self_att_blocks(sum)      # (B, T, C = NB_EMB_DIM)

        logits = self.lang_model_head(pre_logits) # (B, T, C = VOCAB_SIZE)
        return logits

    def loss(self, x, y):
        logits = self.forward(x)
        B, T, C = logits.shape #B = BATCH_SIZE, T = BLOCK_SIZE, C = VOCAB_SIZE
        logits = logits.view(B * T, C)
        y = y.view(B * T)
        logits.shape, y.shape
        return F.cross_entropy(logits, y)
    
    def generate(self, x, nb_new):
        for _ in range(nb_new):
            # x.shape = (B, T)
            x_crop = x[:, -BLOCK_SIZE:] # crop x to the last BLOCK_SIZE tokens
            logits = self.forward(x_crop) # shape (B, T, C)
            logits = logits[:,-1,:]  # shape (B, C)
            probs = F.softmax(logits, dim = -1) # shape (B, C)
            next_x = torch.multinomial(input = probs, num_samples = 1) # shape (B, 1)
            x = torch.cat((x, next_x), dim = 1) # shape (B, T + 1)
        return x
###____________________

### Get model and print number of parameters ###
model = GPT()
model = model.to(device)
nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'number parameters= {nb_params}\n')

### Before training ###
x = torch.zeros(size = (1, 1), dtype = torch.int64, device = device) # decode[0] = \n
model.eval()
print('Text generated before training:\n')
with torch.inference_mode():
    print(decode(model.generate(x, nb_new = 500)[0].tolist()))
###____________________

### Training ###
model.train()
optim = torch.optim.Adam(model.parameters(), lr = LR)

for iter in trange(MAX_ITERS):
    xb, yb = get_batch('train')
    loss = model.loss(xb, yb)

    optim.zero_grad(set_to_none = True)
    loss.backward()
    optim.step()

    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss(model = model)
        print(f'step {iter}: train loss {losses['train']:.4f}', f'step {iter}: val loss {losses['eval']:.4f}')

losses = estimate_loss(model = model)
print(f'step {iter}: train loss {losses['train']:.4f}', f'step {iter}: val loss {losses['eval']:.4f}')
###____________________

### After training ###
x = torch.zeros(size = (1, 1), dtype = torch.int64, device = device)
model.eval()
print('Text generated after training:\n')
with torch.inference_mode():
    print(decode(model.generate(x, nb_new = 500)[0].tolist()))
###____________________