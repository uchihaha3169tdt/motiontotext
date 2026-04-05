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
    """Standard encoder layer that stores attention weights for inspection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        try:
            src2, attn = self.self_attn(
                src, src, src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
                is_causal=is_causal,
            )
        except TypeError:
            src2, attn = self.self_attn(
                src, src, src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
        self.attention_weights = attn.detach()
        src = self.norm1(src + self.dropout1(src2))
        src = self.norm2(src + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(src))))))
        return src


class CSLRTransformer(nn.Module):
    def __init__(self, num_classes, input_dim=231, d_model=512, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.pose_embed = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding1D(d_model)

        def _make_encoder():
            return nn.TransformerEncoder(
                TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True),
                num_layers=num_layers,
            )

        self.enc1 = _make_encoder()
        self.enc2 = _make_encoder()
        self.enc3 = _make_encoder()
        self.enc4 = _make_encoder()

        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.tcn1  = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.tcn2  = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        print(f"CSLRTransformer | input_dim={input_dim} | num_classes={num_classes} | d_model={d_model}")

    def forward(self, poses):
        # poses: (B, T, J, D) or (B, T, D)
        B, T = poses.shape[:2]
        x = poses.view(B, T, -1)          # (B, T, input_dim)
        x = self.pose_embed(x)             # (B, T, d_model)
        x = x + self.pos_enc(x)

        x = self.enc1(x)
        x = self.enc2(x) + x
        x = self.enc3(x) + x
        x = self.enc4(x) + x

        # temporal downsampling: T -> T/4
        x = x.transpose(1, 2)             # (B, d_model, T)
        x = self.pool1(x)
        x = torch.relu(self.tcn1(x))
        x = self.pool2(x)
        x = torch.relu(self.tcn2(x))
        x = x.transpose(1, 2)             # (B, T/4, d_model)

        return self.fc(x)                  # (B, T/4, num_classes)