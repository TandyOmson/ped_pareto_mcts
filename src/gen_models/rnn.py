import torch
import torch.nn as nn
import torch.nn.functional as F

class Vocab:
    def __init__(self, tokens, pad="<pad>", unk="<unk>"):
        uniq = list(dict.fromkeys(tokens))
        self.pad = pad
        self.unk = unk
        self.itos = [pad, unk] + uniq
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @property
    def size(self):
        return len(self.itos)

    def encode(self, seq):
        return [self.stoi.get(t, self.stoi[self.unk]) for t in seq]

    def decode(self, ids):
        return [self.itos[i] for i in ids]

class GRUNextTokenLM(nn.Module):
    def __init__(self, vocab: list, embed_dim=128, hidden_dim=256, num_layers=1):
        super().__init__()
        self.vocab = Vocab(vocab)
        self.pad_idx = self.vocab.stoi[self.vocab.pad]

        self.emb = nn.Embedding(self.vocab.size, embed_dim, padding_idx=self.pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, self.vocab.size)

    def forward_logits(self, x):
        # x: [B, T] token IDs
        emb = self.emb(x)
        h, _ = self.gru(emb)
        return self.proj(h)          # [B, T, V]

    def forward(self, seqs):
        """
        seqs: List[List[str]]
        returns: probs of next token [B, V]
        """
        ids = [self.vocab.encode(s) for s in seqs]
        x = torch.tensor(ids, dtype=torch.long)
        logits = self.forward_logits(x)
        return F.softmax(logits[:, -1], dim=-1)

    @torch.no_grad()
    def get_all_prob_next_symbol(self, seq, temperature=1.0):
        """
        seq: List[str]
        returns: Dict[str, float]  (token -> probability)
        """
        self.eval()
        ids = torch.tensor([self.vocab.encode(seq)], dtype=torch.long)
        logits = self.forward_logits(ids)[:, -1] / temperature
        probs = F.softmax(logits, dim=-1)[0]
        return {self.vocab.itos[i]: float(probs[i]) for i in range(self.vocab.size)}
    
def fit(
    model: GRUNextTokenLM,
    sequences,              # List[List[str]]
    epochs=10,
    lr=1e-3,
):
    vocab = model.vocab
    pad = vocab.stoi[vocab.pad]

    # encode
    encoded = [vocab.encode(s) for s in sequences]
    max_len = max(len(s) for s in encoded)

    # pad
    x = torch.full((len(encoded), max_len), pad, dtype=torch.long)
    for i, s in enumerate(encoded):
        x[i, :len(s)] = torch.tensor(s)

    # teacher forcing
    inputs  = x[:, :-1]
    targets = x[:, 1:]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad)

    model.train()
    for ep in range(1, epochs + 1):
        logits = model.forward_logits(inputs)
        loss = loss_fn(
            logits.reshape(-1, vocab.size),
            targets.reshape(-1),
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"epoch {ep:03d} | loss = {loss.item():.4f}")

if __name__ == "__main__":
    import argparse
    from gen_models.model_utils import tokenize_smiles, save_model, save_tokens
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles")
    parser.add_argument("--model_out")
    parser.add_argument("--vocab_out")
    
    args = parser.parse_args()

    smis = [i.strip() for i in open(Path(args.smiles)).readlines()]
    all_tokens, sequences = tokenize_smiles(smis)

    model = GRUNextTokenLM(all_tokens, embed_dim=64, hidden_dim=128)
    fit(model, sequences, epochs=5)

    # probs = model.get_all_prob_next_symbol(["C", "C"])
    # print(probs)

    save_model(model, Path(args.model_out))
    save_tokens(all_tokens + "<unk>" + '$', Path(args.vocab_out))
