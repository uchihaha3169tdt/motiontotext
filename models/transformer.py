import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        B, T, _ = x.shape
        t = torch.arange(T, device=x.device).float()
        sin_inp = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.zeros(T, self.channels, device=x.device, dtype=x.dtype)
        emb[:, 0::2] = sin_inp.sin()
        emb[:, 1::2] = sin_inp.cos()
        return emb[None, :, :x.shape[-1]].expand(B, -1, -1)


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src_norm = self.norm1(src)
        try:
            src2, attn = self.self_attn(
                src_norm, src_norm, src_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
                is_causal=is_causal,
            )
        except TypeError:
            src2, attn = self.self_attn(
                src_norm, src_norm, src_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
        self.attention_weights = attn.detach()
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
        src = src + self.dropout2(src2)
        return src


class CSLRTransformer(nn.Module):
    """
    Transformer CSLR với CTC.
    Downsampling T -> T/2 (một pool duy nhất).
    """
    def __init__(self, num_classes, input_dim=231, d_model=256, nhead=4,
                 num_layers=2, dropout=0.1):
        super().__init__()

        self.pose_embed = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc    = PositionalEncoding1D(d_model)
        self.input_drop = nn.Dropout(p=0.1)

        def _make_encoder():
            return nn.TransformerEncoder(
                TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=False,
                ),
                num_layers=num_layers,
            )

        self.enc1 = _make_encoder()
        self.enc2 = _make_encoder()
        self.enc3 = _make_encoder()
        self.enc4 = _make_encoder()

        self.temporal_pool = nn.AvgPool1d(kernel_size=2, stride=2)

        self.tcn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()
        print(f"CSLRTransformer | input_dim={input_dim} | num_classes={num_classes} "
              f"| d_model={d_model} | nhead={nhead} | pool=T/2")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, poses):
        B, T = poses.shape[:2]
        x = poses.view(B, T, -1)
        x = self.pose_embed(x)
        x = self.input_drop(x + self.pos_enc(x))
        x = self.enc1(x)
        x = self.enc2(x) + x
        x = self.enc3(x) + x
        x = self.enc4(x) + x
        x = x.transpose(1, 2)
        x = self.temporal_pool(x)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        return self.fc(x)   # (B, T/2, num_classes)