from typing import Sequence, Tuple
import jax.numpy as jnp
from flax import linen as nn
from .embed import TokenEmbedding
from .head import MLP


def causal_mask(T: int) -> jnp.ndarray:
    return jnp.tril(jnp.ones((T, T), dtype=bool))

class PositionalEmbedding(nn.Module):
    max_len: int
    d_model: int

    @nn.compact
    def __call__(self, T: int) -> jnp.ndarray:
        pos_ids = jnp.arange(T, dtype=jnp.int32)  # (T,)
        return nn.Embed(num_embeddings=self.max_len, features=self.d_model)(pos_ids)  # (T, d_model)

class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Masked self-attention 
        y = nn.LayerNorm()(x)
        T = y.shape[0]
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
        )(y, y, mask=causal_mask(T))  # (T, d_model)
        
        # Residual connection
        x = x + y 

        # FFN 
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.d_ff)(y)   # (T, d_ff)
        y = nn.relu(y)
        y = nn.Dense(self.d_model)(y)  # (T, d_model)
        
        # Residual connection
        x = x + y 

        return x

class DecoderOnlyTransformer(nn.Module):
    vocab_size: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    max_len: int
    mlp_widths: Sequence[int]  

    @nn.compact
    def __call__(self, x_ids: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Token embedding
        x = TokenEmbedding(vocab_size=self.vocab_size, embed_dim=self.d_model)(x_ids)

        # Positional embedding
        T = x.shape[0]
        x = x + PositionalEmbedding(max_len=self.max_len, d_model=self.d_model)(T)

        # Transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
            )(x)

        # Norm Layer
        x = nn.LayerNorm()(x)

        # MLP head
        logits = MLP(self.mlp_widths)(x)

        return logits, x
