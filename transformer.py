import torch, copy, math
import torch.nn.functional as F 
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

    def decoder(self, encoded, src_mask, tar, tar_mask):
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
        self.norm = LayerNorm(Layer.size)

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
        std = x.std(axis-1, keepdims=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


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

    def forward(self, query, key, value, mask):
        nbatches = x.size(0)

        query, key, value = [l(x).view(nbatches, -1, nheads, dk).transpose(1, 2)
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
        self.dmodel = dmodel

    def forward(self, x):
        return self.l(x) * math.sqrt(self.dmodel)

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
        nn.Sequential(Embedding(dmodel, target_emb), c(pos)),
        Generator(dmodel, target_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model


