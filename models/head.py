from typing import Sequence
import jax
import jax.numpy as jnp
from flax import linen as nn


# MLP
class MLP(nn.Module):
    widths: Sequence[int]  # np. [128, 128, C]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, w in enumerate(self.widths):
            x = nn.Dense(w)(x)
            if i < len(self.widths) - 1:
                x = nn.relu(x)
        return x  

