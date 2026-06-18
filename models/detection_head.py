import math
import torch
import torch.nn as nn


class SinCos2DPositionalEncoding(nn.Module):
    """2D sinusoidal positional encoding appended to spatial feature tokens."""

    def __init__(self, channels, temperature=10000):
        super().__init__()
        assert channels % 4 == 0, "channels must be divisible by 4 for 2D sin/cos encoding"
        self.channels = channels
        self.temperature = temperature

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        y_pos = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(1).expand(H, W)
        x_pos = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0).expand(H, W)

        dim = C // 4
        omega = torch.arange(dim, device=device, dtype=torch.float32)
        omega = 1.0 / (self.temperature ** (omega / dim))

        y_enc = torch.einsum('hw,d->hwd', y_pos, omega)
        x_enc = torch.einsum('hw,d->hwd', x_pos, omega)

        pos = torch.cat([y_enc.sin(), y_enc.cos(), x_enc.sin(), x_enc.cos()], dim=-1)
        return pos.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)


class TransformerDetectionHead(nn.Module):
    """
    DETR-style detection head.
    Learned query embeddings attend to multi-scale FPN memory tokens
    to predict bounding boxes and defect class labels.
    """

    def __init__(self, d_model=256, num_queries=100, num_classes=6,
                 num_decoder_layers=6, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        self.pos_enc = SinCos2DPositionalEncoding(d_model)
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.input_proj = nn.Conv2d(d_model, d_model, kernel_size=1)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model)
        )

        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1 = no-object
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.bbox_head[-2].bias.data[:2], -0.5, 0.5)
        nn.init.constant_(self.bbox_head[-2].bias.data[2:], -2.0)

    def forward(self, fpn_features):
        B = fpn_features[0].shape[0]

        # Flatten all FPN levels into one memory sequence
        tokens = []
        for feat in fpn_features:
            feat = self.input_proj(feat)
            pos = self.pos_enc(feat)
            tokens.append((feat + pos).flatten(2).permute(0, 2, 1))  # [B, HW, C]

        memory = torch.cat(tokens, dim=1)  # [B, sum_HW, C]

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, Q, C]
        out = self.decoder(queries, memory)  # [B, Q, C]

        return {
            'pred_logits': self.class_head(out),   # [B, Q, num_classes+1]
            'pred_boxes': self.bbox_head(out),      # [B, Q, 4]  (cx,cy,w,h) normalized
        }
