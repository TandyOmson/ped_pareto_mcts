import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from tqdm import trange

class Vocab:    
    def __init__(self, tokens, pad="<pad>", unk="<unk>", bos="<bos>", eos="<eos>"):
        self.pad, self.unk, self.bos, self.eos = pad, unk, bos, eos
        base = [t for t in tokens if t not in (pad, unk, bos, eos)]
        self.itos = [pad, unk, bos, eos] + base
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @property
    def size(self):
        return len(self.itos)

    def encode(self, seq):
        return [self.stoi.get(t, self.stoi[self.unk]) for t in seq]

    def decode(self, ids):
        return [self.itos[i] for i in ids]

class TransformerNextTokenLM(nn.Module):
    """ MolGPT-like causal transformer decoder for next token prediction
    """
    def __init__(self, 
                 all_tokens,
                 d_model=256, 
                 nhead=8, 
                 num_layers=6, 
                 dim_feedforward=1024, 
                 dropout=0.1, 
                 max_len=30
                 ):
        super().__init__()

        self.vocab = Vocab(all_tokens)
        self.vocab_size = self.vocab.size
        self.pad_idx = self.vocab.stoi[self.vocab.pad]
        self.max_len = max_len
        self.d_model = d_model

        self.tok_emb = nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_idx)
        self.pos_emb = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False) 

    def _causal_mask(self, T, device):
        # True masks the item going into the transformer
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward_logits(self, x, state=None):
        # x: [batch_size, seq_length]
        B, T = x.shape
        device = x.device

        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        h = self.tok_emb(x) + self.pos_emb(pos)

        attn_mask = self._causal_mask(T, device)
        key_padding_mask = (x == self.pad_idx)

        h = self.transformer_encoder(h, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        logits =  self.lm_head(h) # [batch_size, seq_length, vocab_size]
        return logits, None

    @torch.no_grad()
    def get_all_prob_next_symbol(self, seq, temperature=1.0):
        """
        For integration in MCTS
        seq: single sequence
        returns: dictionary (token -> probability)
        """
        self.eval()
        device = next(self.parameters()).device
        ids = torch.tensor([self.vocab.encode(seq)], dtype=torch.long, device=device)
        logits, _ = self.forward_logits(ids)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)[0]
        return {self.vocab.itos[i]: float(probs[i]) for i in range(self.vocab.size)}
    
class TransformerTrainer():
    @staticmethod
    def fit(
        model: TransformerNextTokenLM,
        sequences,              
        epochs=250,
        lr=1e-3,
    ):
        vocab = model.vocab
        pad = vocab.stoi[vocab.pad]
        bos = vocab.stoi[vocab.bos]
        eos = vocab.stoi[vocab.eos]

        # encode
        encoded = [[bos] + vocab.encode(s) + [eos] for s in sequences]
        max_len = model.max_len

        # pad
        x = torch.full((len(encoded), max_len), pad, dtype=torch.long)
        for i, s in enumerate(encoded):
            x[i, :len(s)] = torch.tensor(s)

        # teacher forcing (i.e. training on all possible unfinished sequences)
        inputs  = x[:, :-1]
        targets = x[:, 1:]

        # training is currently fully teacher forced, could also predict multiple steps ahead or use scheduled sampling (gradually reduce teacher forcing)

        device = next(model.parameters()).device
        inputs, targets = inputs.to(device), targets.to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad)

        model.train()
        with trange(epochs) as t:
            for ep in t:
                t.set_description('Training RNN: epoch %d' % (ep + 1))
                logits, _ = model.forward_logits(inputs)
                loss = loss_fn(
                    logits.reshape(-1, vocab.size),
                    targets.reshape(-1),
                )

                opt.zero_grad()
                loss.backward()
                opt.step()
                t.set_postfix(epoch=f"{ep:03d}", loss=f"{loss.item():.4f}")
