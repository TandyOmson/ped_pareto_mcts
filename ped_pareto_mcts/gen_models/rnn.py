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

class LSTMNextTokenLM(nn.Module):
    """ MolGPT-like causal transformer decoder for next token prediction
    """
    def __init__(self, vocab: list, embed_dim=128, hidden_dim=256, max_len=30):
        super().__init__()
        self.vocab = Vocab(vocab)
        self.pad_idx = self.vocab.stoi[self.vocab.pad]
        self.max_len = max_len

        # embed token IDs to vectors, padding token does not contribute to gradients
        self.emb = nn.Embedding(self.vocab.size, embed_dim, padding_idx=self.pad_idx)

        # output hidden states
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # project hidden states to vocab size for next token prediction
        self.proj = nn.Linear(hidden_dim, self.vocab.size)

    def forward_logits(self, x, state=None):
        # x: [batch_size, seq_len] token IDs
        lengths = (x != self.pad_idx).sum(dim=1)
        lengths_cpu = lengths.clamp(min=1).cpu()

        emb = self.emb(x)
        packed_emb = rnn_utils.pack_padded_sequence(emb, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_out, new_state = self.lstm(packed_emb, state)
        out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        logits = self.proj(out)  # [batch_size, seq_len, vocab_size]
        return logits, new_state

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
    
    @torch.no_grad()
    def step(self, token_idx, state=None, temperature=1.0):
        """ 
        Alternative integration into MCTS
        Requires caching hidden states in nodes (memory tradeoff, but faster inference)
        """
        self.eval()
        device = next(self.parameters()).device
        ids = torch.as_tensor([[token_idx]], dtype=torch.long, device=device)
        logits, new_state = self.forward_logits(ids, state=state)
        logits = logits[:, -1] / max(temperature, 1e-8)
        probs = F.softmax(logits, dim=-1)[0]
        return probs, new_state
    
class LSTMTrainer():

    @staticmethod
    def fit(
        model: LSTMNextTokenLM,
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
        max_len = max(len(s) for s in encoded)

        # pad
        x = torch.full((len(encoded), max_len), pad, dtype=torch.long)
        for i, s in enumerate(encoded):
            x[i, :len(s)] = torch.tensor(s)

        # teacher forcing (i.e. training on all possible unfinished sequences)
        inputs  = x[:, :-1]
        targets = x[:, 1:]

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
