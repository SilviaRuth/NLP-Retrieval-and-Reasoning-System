from __future__ import annotations

import torch
import torch.nn as nn


class LSTMNLIClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        num_labels: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.encoder = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_idx).long()

        embedded = self.embedding(input_ids)
        encoded, _ = self.encoder(embedded)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)


class CNNNLIClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        num_labels: int,
        embed_dim: int = 128,
        num_filters: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, num_filters, kernel_size=kernel)
                for kernel in (3, 4, 5)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_filters * len(self.convs), num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        del attention_mask
        embedded = self.embedding(input_ids).transpose(1, 2)
        conv_features = []
        for conv in self.convs:
            activated = torch.relu(conv(embedded))
            pooled = torch.max(activated, dim=2).values
            conv_features.append(pooled)
        features = torch.cat(conv_features, dim=1)
        return self.classifier(features)
