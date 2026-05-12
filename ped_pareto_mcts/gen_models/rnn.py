import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from ped_pareto_mcts.base.gen_model import GenModel

class LSTMNextTokenLM(GenModel):
    """ MolGPT-like causal transformer decoder for next token prediction
    """
    def __init__(self, 
                 all_tokens, 
                 embed_dim=128, 
                 max_len=30, 
                 hidden_dim=256
                 ):
        super().__init__(all_tokens, embed_dim, max_len)

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
