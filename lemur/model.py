from __future__ import annotations

import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=1024,
        final_hidden_dim=None,
        num_layers=2,
        activation="relu",
    ):
        super().__init__()

        if final_hidden_dim is None:
            final_hidden_dim = hidden_dim

        self.config = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
            "final_hidden_dim": final_hidden_dim,
            "num_layers": num_layers,
            "activation": activation,
        }

        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        if activation not in activations:
            raise ValueError("activation must be one of: relu, gelu, silu")
        activation_cls = activations[activation]

        modules = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        for i, in_dim in enumerate(dims):
            is_last = i == len(dims) - 1
            out_dim = final_hidden_dim if is_last else hidden_dim

            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(nn.LayerNorm(out_dim))
            modules.append(activation_cls())

        self.feature_extractor = nn.Sequential(*modules)

        self.output_layer = nn.Linear(final_hidden_dim, output_dim, bias=False)

    def forward(self, x):
        feats = self.feature_extractor(x)
        return self.output_layer(feats)