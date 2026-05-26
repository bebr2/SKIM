from __future__ import annotations

import torch
from torch import nn


class MLPProjector(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        layers: int = 3,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")

        dtype = torch_dtype if torch_dtype is not None else torch.float32

        modules: list[nn.Module] = []
        if layers == 1:
            linear = nn.Linear(in_dim, out_dim)
            linear.weight = nn.Parameter(linear.weight.to(dtype))
            if linear.bias is not None:
                linear.bias = nn.Parameter(linear.bias.to(dtype))
            modules.append(linear)
        else:
            # first layer
            linear1 = nn.Linear(in_dim, hidden_dim)
            linear1.weight = nn.Parameter(linear1.weight.to(dtype))
            if linear1.bias is not None:
                linear1.bias = nn.Parameter(linear1.bias.to(dtype))
            modules.append(linear1)
            modules.append(nn.GELU())
            # middle layers
            for _ in range(layers - 2):
                linear_mid = nn.Linear(hidden_dim, hidden_dim)
                linear_mid.weight = nn.Parameter(linear_mid.weight.to(dtype))
                if linear_mid.bias is not None:
                    linear_mid.bias = nn.Parameter(linear_mid.bias.to(dtype))
                modules.append(linear_mid)
                modules.append(nn.GELU())
            # last layer
            linear_last = nn.Linear(hidden_dim, out_dim)
            linear_last.weight = nn.Parameter(linear_last.weight.to(dtype))
            if linear_last.bias is not None:
                linear_last.bias = nn.Parameter(linear_last.bias.to(dtype))
            modules.append(linear_last)
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
