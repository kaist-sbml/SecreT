import math
import torch
import torch.nn as nn
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)


    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class SecreTransformer(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_p):
        super().__init__()
        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=5000)
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
        d_model=dim_model,
        nhead=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)


    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)
        return out


    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask


    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        return (matrix == pad_token)
    
def beam_search_with_threshold(model, src, start_token, beam_width, max_len, prob_threshold, temperature, device):
    model.eval()
    with torch.no_grad():
        src = model.embedding(src) * torch.sqrt(torch.tensor(model.dim_model, dtype=torch.float))
        src = model.positional_encoder(src).permute(1, 0, 2)
        memory = model.transformer.encoder(src)

        beams = [(torch.tensor([[start_token]], device=device), 0.0)]
        final_results = []

        for _ in tqdm(range(1, max_len + 1)):
            new_beams = []
            for tgt, score in beams:
                tgt_emb = model.embedding(tgt) * torch.sqrt(torch.tensor(model.dim_model, dtype=torch.float))
                tgt_emb = model.positional_encoder(tgt_emb).permute(1, 0, 2)
                tgt_mask = model.get_tgt_mask(tgt.shape[1]).to(device)
                output = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                logits = model.out(output[-1]) / temperature
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                valid_mask = probs >= prob_threshold
                valid_probs = probs[valid_mask]
                valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
                if valid_probs.numel() == 0:
                    continue
                top_probs, top_indices = torch.topk(valid_probs, min(beam_width, len(valid_probs)))
                top_tokens = valid_indices[top_indices]
                for i in range(len(top_tokens)):
                    new_token = top_tokens[i].item()
                    new_score = score + torch.log(top_probs[i]).item()
                    new_tgt = torch.cat([tgt, torch.tensor([[new_token]], device=device)], dim=-1)
                    if new_token == 8:
                        final_results.append((new_tgt, new_score))
                    else:
                        new_beams.append((new_tgt, new_score))
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            if not new_beams and final_results:
                break


        if not final_results:
            final_results = beams
        final_results = sorted(final_results, key=lambda x: x[1], reverse=True)[:beam_width]
        return [seq.tolist()[0] for seq, _ in final_results]