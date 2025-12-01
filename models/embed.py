import jax.numpy as jnp
from flax import linen as nn

class TokenEmbedding(nn.Module):
    vocab_size: int   
    embed_dim: int    

    @nn.compact
    def __call__(self, x_ids: jnp.ndarray) -> jnp.ndarray:
        x_ids = x_ids.astype(jnp.int32)
        emb = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embed_dim,
            dtype=jnp.float32,
        )
        return emb(x_ids)
