from typing import Sequence, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import lax
from flax import linen as nn
from .head import MLP
from .embed import TokenEmbedding

# Linear Recurrent Unit (LRU)
class LinearRecurrentUnit(nn.Module):
    hidden_dim: int         # H - wymiar stanu h_t
    input_dim: int          # X - wymiar wejścia x_t
    a_scale: float = 0.1
    a_identity: float = 0.9
    parallel_scan: bool = True 

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
            h0 = jnp.zeros((H,), dtype=x_seq.dtype)

        if not self.parallel_scan:
            def step(h, x_t):
                # (H,) = (H,H) x (H,) + (H,X) x (X,) + (H,)
                h_new = A @ h + B @ x_t + c
                return h_new, h_new
            _, h_all = lax.scan(step, h0, x_seq)  # (T,H)
            return h_all

        # b_t = B x_t + c -> (T,H,1)
        b_seq = (x_seq @ B.T + c[None, :])[..., None] 
        A_seq = jnp.broadcast_to(A, (T, H, H))  # (T,H,H)

        def combine(left, right):
            A1, b1 = left  # (H,H), (H,1)
            A2, b2 = right  # (H,H), (H,1)
            return (A2 @ A1, A2 @ b1 + b2)

        A_pref, b_pref = lax.associative_scan(combine, (A_seq, b_seq), axis=0)  # (T,H,H), (T,H,1)

        b_pref = b_pref[..., 0] 
        h_all = (A_pref @ h0) + b_pref  # (T,H)

        return h_all


class ScanSequenceModel(nn.Module):
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
        
        h_all = LinearRecurrentUnit(
            hidden_dim=self.hidden_dim,
            input_dim=self.embed_dim,
            parallel_scan=True
        )(x_emb, h0)                  

        y_all = MLP(self.mlp_widths)(h_all)     

        return y_all, h_all
