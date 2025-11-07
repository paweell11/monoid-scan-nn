import jax
import jax.numpy as jnp
import optax
import time

from models.lru import ScanSequenceModel
from data import CharData, make_window_starts


BATCH_SIZE = 16
SEQ_LEN = 128           # T
HIDDEN_DIM = 256        # H
MLP_HIDDEN = 512
LR = 1e-3
MAX_STEPS = 10000
LOG_EVERY = 400


def setup_model(vocab_size: int, rng):
    model = ScanSequenceModel(
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

def generate(model, params, dm, prompt: str, max_new_tokens: int = 100):
    ids = dm.encode_str(prompt)
    context_ids = list(ids)
    h0 = jnp.zeros((HIDDEN_DIM,))

    for _ in range(max_new_tokens):
        x_ids = jnp.array(context_ids, dtype=jnp.int32) 
        x_oh = jax.nn.one_hot(x_ids, dm.vocab_size())
        logits, _ = model.apply(params, x_oh, h0)
        last_logit = logits[-1]
        next_id = int(jnp.argmax(last_logit))
        context_ids.append(next_id)
    
    return dm.decode_ids(context_ids)

def main():
    dm = CharData()
    dm.prepare()

    V = dm.vocab_size()
    print(f"Vocab size: {V}")
    print(f"Train tokens: {len(dm.train_ids)}")
    num_windows = len(make_window_starts(len(dm.train_ids), SEQ_LEN))
    steps_per_epoch = num_windows // BATCH_SIZE
    epochs = MAX_STEPS // steps_per_epoch
    print(f"Step per epoch: {steps_per_epoch}")
    print(f"Number of epochs: {epochs}")

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
            sec_per_step = dt / LOG_EVERY
            steps_per_sec = LOG_EVERY / dt
            chars_per_sec = LOG_EVERY * BATCH_SIZE * SEQ_LEN / dt
            print(f"[step {step:5d}] | train_loss = {float(loss):.4f} | val_loss = {float(val_loss):.4f} | "
                f"ppl={val_ppl:.3f} | bpc={val_bpc:.3f} | sec/step={sec_per_step:.4f} | steps/sec={steps_per_sec:.2f} | "
                f"chars/sec={chars_per_sec:.2f} ({dt:.1f}s)"
            )
            t0 = time.time()
    try:
        prompt = "Litwo! Ojczyzny moja"
        sample = generate(model, params, dm, prompt=prompt, max_new_tokens=100)
        print(f"\n=== SAMPLE for {prompt} ===")
        print(sample)
    except Exception as e:
        print("Sampling failed:", e)


if __name__ == "__main__":
    main()
