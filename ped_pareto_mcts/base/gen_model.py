from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import trange

class Vocab:    
    """ Universal vocabulary class for tokenization and encoding/decoding
    """
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

class GenModel(ABC, torch.nn.Module):
    """ ABC for autoregressive generative models
    """
    def __init__(self, all_tokens: list, embed_dim: int, max_len: int):
        super().__init__()
        self.vocab = Vocab(all_tokens)
        self.pad_idx = self.vocab.stoi[self.vocab.pad]
        self.embed_dim = embed_dim
        self.max_len = max_len

    def forward_logits(self, x, state=None):
        """ Foward pass for next token prediction
        """
        return NotImplementedError
    
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

class GenModelTrainer:
    """ Generic trainer for autoregressive generative models
        Currently only applied teacher forcing, can add other methods later
    """
    @staticmethod
    def fit(
        model: GenModel,
        sequences : list,              
        epochs : int = 250,
        train_config: dict = None,
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

        opt = torch.optim.Adam(model.parameters(), lr=train_config.get("lr", 1e-3))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=train_config.get("lr_decay", 0.00))
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
                scheduler.step()
                t.set_postfix(loss=f"{loss.item():.4f}")


    @staticmethod
    def fit_scheduled_sampling(
        model: GenModel,
        sequences: list,
        epochs: int = 250,
        train_config: dict = None,
    ):
        """ Schedule sampling initially uses teacher forcing, gradually starts using model predictions (hill climbing style)
            This means that ground truth are no longer inputs, but fixed instability at start of hill climbing
        """
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

        inputs  = x[:, :-1]
        targets = x[:, 1:]

        device = next(model.parameters()).device
        inputs, targets = inputs.to(device), targets.to(device)

        opt = torch.optim.Adam(model.parameters(), lr=train_config.get("lr", 1e-3))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=train_config.get("lr_decay", 0.00))
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad)

        model.train()
        with trange(epochs) as t:
            for ep in t:
                t.set_description('Training RNN: epoch %d' % (ep + 1))
                # Scheduled sampling: at timestep t, choose input as ground truth or model prediction increasing probabiliy of model prediction
                teacher_decay_rate = train_config.get("teacher_decay", 0.95)
                teacher_prob = max(0.1, teacher_decay_rate ** ep)
                
                logits, _ = model.forward_logits_scheduled(inputs, teacher_prob)
                loss = loss_fn(
                    logits.reshape(-1, vocab.size),
                    targets.reshape(-1),
                )

                opt.zero_grad()
                loss.backward()
                opt.step()
                scheduler.step()
                t.set_postfix(loss=f"{loss.item():.4f}")