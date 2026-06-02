from __future__ import annotations

import math

import torch
from torch import Tensor


def hippo_legs_matrix(size: int, dtype: torch.dtype = torch.float32) -> Tensor:
    matrix = torch.zeros(size, size, dtype=dtype)
    for n in range(size):
        for k in range(size):
            if n > k:
                matrix[n, k] = -math.sqrt((2 * n + 1) * (2 * k + 1))
            elif n == k:
                matrix[n, k] = -(n + 1)
    return matrix
