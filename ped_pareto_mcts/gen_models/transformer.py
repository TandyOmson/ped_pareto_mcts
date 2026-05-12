import torch
import torch.nn as nn

from ped_pareto_mcts.base.gen_model import GenModel

class TransformerNextTokenLM(GenModel):
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
        super().__init__(all_tokens, d_model, max_len)

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
