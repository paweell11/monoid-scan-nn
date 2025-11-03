import jax
import jax.numpy as jnp

from model import SequenceModel


BATCH_SIZE = 16
SEQ_LEN = 128           # T
HIDDEN_DIM = 256        # H
MLP_HIDDEN = 512
LR = 1e-3
MAX_STEPS = 10000
LOG_EVERY = 400


def setup_model(vocab_size: int, rng):
    model = SequenceModel(
        input_dim=vocab_size, hidden_dim=HIDDEN_DIM, mlp_widths=[MLP_HIDDEN, vocab_size],  
    )
    x_dummy = jnp.zeros((SEQ_LEN, vocab_size), jnp.float32)
    h0_dummy = jnp.zeros((HIDDEN_DIM,), jnp.float32)
    params = model.init(rng, x_dummy, h0_dummy)  
    return model, params
