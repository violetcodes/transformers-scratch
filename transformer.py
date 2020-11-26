import torch, copy, math, time
import torch.nn.functional as F 
from torch.autograd import Variable
import torch.nn as nn 
import numpy as np

class EncoderDecoder(nn.Module):
    def __init__(self, enc, dc, source_emb, target_emb, gen):
        super(EncoderDecoder, self).__init__()
        self.encoder = enc
        self.decoder = dc
        self.source_emb = source_emb
        self.target_emb = target_emb
        self.generator = gen 

    def forward(self, src, src_mask, tar, tar_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tar, tar_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.source_emb(src), src_mask)

    def decode(self, encoded, src_mask, tar, tar_mask):
        return self.decoder(encoded, self.target_emb(tar), src_mask, tar_mask)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
    
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(features))
        self.b1 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b1


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, mem, x, src_mask, tar_mask):
        for layer in self.layers:
            x = layer(mem, x, src_mask, tar_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, enc_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.enc_attn = enc_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, mem, x, src_mask, tar_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tar_mask))
        x = self.sublayer[1](x, lambda x: self.enc_attn(x, mem, mem, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask_ = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask_) == 0

def attention(query, key, value, mask=None, dropout=None):
    d = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d)
    if mask is not None:
        print('score size', scores.shape, 'mask shape', mask.shape)
        scores = scores.masked_fill(mask==0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, dmodel, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert dmodel%h == 0
        self.dk = dmodel//h
        self.h = h
        self.linears = clones(nn.Linear(dmodel, dmodel), 4)
        self.attn = None
        self.dropout = dropout

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        if mask is not None: mask = mask.unsqueeze(1)

        query, key, value = [l(x).view(nbatches, -1, self.h, self.dk).transpose(1, 2)
        for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h*self.dk)
        return self.linears[-1](x)
    

class FeedForward(nn.Module):
    def __init__(self, dmodel, dff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(dmodel, dff)
        self.w2 = nn.Linear(dff, dmodel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class Embedding(nn.Module):
    def __init__(self, dmodel, vocab):
        super(Embedding, self).__init__()
        self.l = nn.Embedding(vocab, dmodel)
        self.d_model = dmodel

    def forward(self, x):
        return self.l(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, dmodel, dropout, maxlen=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(maxlen, dmodel)
        position = torch.arange(0, maxlen).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, dmodel, 2) * (4/dmodel))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x  = x + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
    


def make_model(
    source_vocab,
    target_vocab,
    N=6,
    dmodel=512,
    dff=2048,
    h=8,
    dropout=0.1
):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, dmodel)
    ff = FeedForward(dmodel, dff, dropout)
    pos = PositionalEncoding(dmodel, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(dmodel, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(dmodel, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embedding(dmodel, source_vocab), c(pos)),
        nn.Sequential(Embedding(dmodel, target_vocab), c(pos)),
        Generator(dmodel, target_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model

class Batch:
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask  = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
        
    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(
            tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_loss, total_tokens, tokens = 0, 0, 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        tokens += batch.ntokens
        total_tokens += batch.ntokens
        if i%50 == 1:
            elapsed = time.time() - start
            print(f"Epoch Step:{i} Loss:{loss/batch.ntokens} tokens per second {tokens/elapsed}")
            start = time.time()
            tokens = 0
    return total_loss/total_tokens


global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate 
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(
            step ** (-0.5), step * self.warmup ** (-1.5)
        ))
    
def get_std_opt(model):
    return NoamOpt(
        model.src_embed[0].d_model, 2, 4000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None


    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()

        true_dist.fill_(self.smoothing/(self.size-2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


def data_gen(V, batch, nbatches):
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        trg = Variable(data, requires_grad=False)

        yield Batch(src, trg, 0)


class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
    
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(
            x.contiguous().view(-1, x.size(-1)),
            y.contiguous().view(-1)) / norm
        
        loss.backward()

        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss.data[0] * norm


V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(
    model.source_emb[0].d_model, 1, 400,
    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
))

for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, 20, 20), model, SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model, SimpleLossCompute(model.generator, criterion, None)))    
