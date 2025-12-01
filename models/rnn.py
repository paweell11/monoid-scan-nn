from typing import Sequence, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import lax
from flax import linen as nn
from .head import MLP
from .embed import TokenEmbedding

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
    vocab_size: int
    hidden_dim: int
    mlp_widths: Sequence[int]  
    embed_dim: int

    @nn.compact
    def __call__(self, x_ids: jnp.ndarray, h0: Optional[jnp.ndarray] = None ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x_emb = TokenEmbedding(
            vocab_size = self.vocab_size,
            embed_dim = self.embed_dim
        )(x_ids)
        
        h_all = RecurrentNeuralNetwork(
            hidden_dim=self.hidden_dim,
            input_dim=self.embed_dim,
        )(x_emb, h0)                  

        y_all = MLP(self.mlp_widths)(h_all)  

        return y_all, h_all
