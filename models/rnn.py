from typing import Sequence, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import lax
from flax import linen as nn
from .head import MLP
from .embed import TokenEmbedding

class RecurrentNeuralNetwork(nn.Module):
    hidden_dim: int         # H
    input_dim: int          # X
    a_scale: float = 0.1
    a_identity: float = 0.9

    def setup(self):
        H = self.hidden_dim
        X = self.input_dim
        
        # Parametry
        self.A_raw = self.param("A_raw", nn.initializers.orthogonal(), (H, H))
        self.A = self.a_identity * jnp.eye(H) + self.a_scale * self.A_raw
        self.B = self.param("B", nn.initializers.lecun_normal(), (H, X))
        self.c = self.param("c", nn.initializers.zeros, (H,))

    def __call__(self, x_seq: jnp.ndarray, h0: jnp.ndarray | None = None) -> jnp.ndarray:
        T, X = x_seq.shape
        H = self.hidden_dim

        if h0 is None:
            h0 = jnp.zeros((H,), dtype=x_seq.dtype)

        def step(h, x_t):
            h_new = self.cell(h, x_t)
            return h_new, h_new

        _, h_all = lax.scan(step, h0, x_seq)    # (T,H)
        return h_all

    def cell(self, h: jnp.ndarray, x_t: jnp.ndarray) -> jnp.ndarray:
        h_new = jnp.tanh(self.A @ h + self.B @ x_t + self.c)
        
        return h_new


class RNNModel(nn.Module):
    vocab_size: int
    hidden_dim: int
    mlp_widths: Sequence[int]
    embed_dim: int

    @nn.compact
    def __call__(self, x_ids: jnp.ndarray, h0: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x_emb = TokenEmbedding(
            vocab_size = self.vocab_size,
            embed_dim = self.embed_dim,
            name = "token_embed"
        )(x_ids)
        
        h_all = RecurrentNeuralNetwork(
            hidden_dim = self.hidden_dim,
            input_dim = self.embed_dim,
            name = "rnn_layer"
        )(x_emb, h0)

        y_all = MLP(self.mlp_widths, name="head_mlp")(h_all)

        return y_all, h_all

    @nn.compact
    def infer_step(self, x_id: jnp.ndarray, h_prev: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x_emb = TokenEmbedding(
            vocab_size = self.vocab_size,
            embed_dim = self.embed_dim,
            name = "token_embed"
        )(x_id)
        
        x_emb = x_emb.squeeze()

        rnn_layer = RecurrentNeuralNetwork(
            hidden_dim = self.hidden_dim,
            input_dim = self.embed_dim,
            name = "rnn_layer"
        )
        h_new = rnn_layer.cell(h_prev, x_emb)

        logits = MLP(self.mlp_widths, name="head_mlp")(h_new)

        return logits, h_new