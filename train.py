import jax
import jax.numpy as jnp
import optax
import time

from model import SequenceModel
from data import CharData


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


def make_train_step(model, vocab_size):
    optimizer = optax.adam(LR)

    def forward(params, x_ids): 
        x_oh = jax.nn.one_hot(x_ids, num_classes=vocab_size, dtype=jnp.float32)     # (B,T,V)
        h0 = jnp.zeros((HIDDEN_DIM,), jnp.float32)

        def run_one(x_oh_seq):
            logits, _ = model.apply(params, x_oh_seq, h0)  # (T,V)
            return logits
        return jax.vmap(run_one, in_axes=0)(x_oh)  # (B,T,V)

    def loss_only(params, x_ids, y_ids):
        logits = forward(params, x_ids)
        return optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y_ids).mean()

    @jax.jit
    def train_step(params, opt_state, x_ids, y_ids):
        loss, grads = jax.value_and_grad(loss_only)(params, x_ids, y_ids)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_step(params, x_ids, y_ids):
        loss = loss_only(params, x_ids, y_ids)
        bpc = loss / jnp.log(2.0)
        perplexity = jnp.exp(loss) 
        return loss, bpc, perplexity

    return optimizer, train_step, eval_step

def main():
    dm = CharData()
    dm.prepare()

    V = dm.vocab_size()
    print(f"Vocab size: {V}")

    train_iter = dm.train_loader(batch_size=BATCH_SIZE, shuffle=True)
    val_iter = dm.val_loader(batch_size=BATCH_SIZE, shuffle=False)

    rng = jax.random.PRNGKey(0)
    model, params = setup_model(V, rng)

    param_count = sum(x.size for x in jax.tree.leaves(params))
    print("Total params:", param_count)

    optimizer, train_step, eval_step = make_train_step(model, V)
    opt_state = optimizer.init(params)

    t0 = time.time()
    for step in range(1, MAX_STEPS + 1):
        x_np, y_np = next(train_iter)            
        x = jnp.array(x_np, jnp.int32)
        y = jnp.array(y_np, jnp.int32)

        params, opt_state, loss = train_step(params, opt_state, x, y)

        if step % LOG_EVERY == 0:
            x_val_np, y_val_np = next(val_iter)
            x_val = jnp.array(x_val_np, jnp.int32)
            y_val = jnp.array(y_val_np, jnp.int32)

            val_loss, val_bpc, val_ppl = eval_step(params, x_val, y_val)

            dt = time.time() - t0
            print(f"[step {step:5d}] train_loss = {float(loss):.4f} val_loss = {float(val_loss):.4f} perplexity={val_ppl:.3f} bpc={val_bpc:.3f} ({dt:.1f}s)")
            t0 = time.time()

if __name__ == "__main__":
    main()
