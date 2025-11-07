from typing import Sequence, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import lax
from flax import linen as nn
from .head import MLP

class RecurrentNeuralNetwork(nn.Module):
    hidden_dim: int         # H – wymiar stanu h_t
    input_dim: int          # X – wymiar wejścia x_t
    a_scale: float = 0.1
    a_identity: float = 0.9

    @nn.compact
    def __call__(self, x_seq: jnp.ndarray, h0: jnp.ndarray | None = None) -> jnp.ndarray:
        T, X = x_seq.shape
        H = self.hidden_dim

        # Parametry
        A_raw = self.param("A_raw", nn.initializers.orthogonal(), (H, H))
        A = self.a_identity * jnp.eye(H) + self.a_scale * A_raw       # (H,H)
        B = self.param("B", nn.initializers.lecun_normal(), (H, X))    # (H,X)
        c = self.param("c", nn.initializers.zeros, (H,))              # (H,)

        if h0 is None:
            h0 = jnp.zeros((H,), dtype = x_seq.dtype)

        def step(h, x_t):
            h_new = jnp.tanh(A @ h + B @ x_t + c)
            return h_new, h_new  
        _, h_all = lax.scan(step, h0, x_seq)    # (T,H)
        return h_all    

class RNNModel(nn.Module):
    input_dim: int
    hidden_dim: int
    mlp_widths: Sequence[int]  # np. [128, 128, C]

    @nn.compact
    def __call__(self, x_seq: jnp.ndarray, h0: Optional[jnp.ndarray] = None ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        h_all = RecurrentNeuralNetwork(
            hidden_dim=self.hidden_dim,
            input_dim=self.input_dim,
        )(x_seq, h0)                  # (T,H)

        y_all = MLP(self.mlp_widths)(h_all)  # (T,C)

        return y_all, h_all
