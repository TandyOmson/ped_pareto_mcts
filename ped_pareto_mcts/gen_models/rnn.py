import torch
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
        packed_out, new_state = self.lstm(packed_emb, state) # if state is not this implicitly resets the initial hidden state
        out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        
        logits = self.proj(out)  # [batch_size, seq_len, vocab_size]
        return logits, new_state
    
    def forward_logits_scheduled(self, inputs, teacher_prob):
        logits_all = []

        h, c = None, None # initial hidden state
        input_t = inputs[:, 0] # initial input always bos
        
        for t in range(inputs.size(1)):
            emb_t = self.emb(input_t).unsqueeze(1) # [B, 1, E]
            
            out_t, (h, c) = self.lstm(emb_t, (h, c) if h is not None else None)
            logits_t = self.proj(out_t.squeeze(1)) # [B, V]
            
            logits_all.append(logits_t)

            if t < inputs.size(1)- 1:
                use_teacher = torch.rand(inputs.size(0), device=inputs.device) < teacher_prob
                # sample from models predictions                
                probs = torch.softmax(logits_t, dim=-1)
                pred_tokens = torch.multinomial(probs, 1).squeeze(1)


                next_input = torch.where(use_teacher, inputs[:, t+1], pred_tokens)
                input_t = next_input

        logits = torch.stack(logits_all, dim=1)
        return logits, (h, c)
