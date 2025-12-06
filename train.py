import jax
import jax.numpy as jnp
import optax
import time
import os, json
from datetime import datetime
from typing import Sequence, Optional
from flax import linen as nn

from models.lru import ScanSequenceModel
from models.rnn import RNNModel
from models.transformer import DecoderOnlyTransformer 

from data import CharData, make_window_starts

from utils.early_stopping import EarlyStopper


BATCH_SIZE = 16
SEQ_LEN = 128           # T
HIDDEN_DIM = 64        # H
MLP_HIDDEN = 96
LR = 1e-3
MAX_STEPS = 10000
LOG_EVERY = 400

USE_TRANSFORMER = True  # False = SanSequenceModel/RNN, True = Transformer
D_MODEL = 24
NUM_LAYERS = 2
NUM_HEADS = 2
D_FF = 4 * D_MODEL
MAX_LEN = SEQ_LEN  


MODEL_NAME = "lru_scan"
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_DIR = os.path.join("artifacts", MODEL_NAME, f"run-{RUN_ID}")
os.makedirs(RUN_DIR, exist_ok=True)
JSONL_PATH = os.path.join(RUN_DIR, "metrics.jsonl")


class TransformerAsSequenceModel(nn.Module):
    vocab_size: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    max_len: int
    mlp_widths: Sequence[int]

    @nn.compact
    def __call__(self, x_ids: jnp.ndarray, h0: Optional[jnp.ndarray] = None):
        logits, hidden = DecoderOnlyTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            max_len=self.max_len,
            mlp_widths=self.mlp_widths,
        )(x_ids)
        return logits, hidden


def setup_model(vocab_size: int, rng, hidden_dim: int, mlp_hidden: int, seq_len: int, d_model: int, 
                num_layers: int, num_heads: int, d_ff: int, max_len: int):
    x_dummy = jnp.zeros((seq_len,), jnp.int32)

    if not USE_TRANSFORMER:
        model = ScanSequenceModel(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            mlp_widths=[mlp_hidden, vocab_size],
            embed_dim=96,
        )
        h0_dummy = jnp.zeros((hidden_dim,), jnp.float32)
        params = model.init(rng, x_dummy, h0_dummy)

    else:
        model = TransformerAsSequenceModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_len=max_len,
            mlp_widths=[mlp_hidden, vocab_size],
        )
        h0_dummy = jnp.zeros((1,), jnp.float32)  
        params = model.init(rng, x_dummy, h0_dummy)

    return model, params

def make_train_step(model, vocab_size: int, hidden_dim: int, lr: float):
    optimizer = optax.adam(lr)

    def forward(params, x_ids): 
        h0 = jnp.zeros((hidden_dim,), jnp.float32)

        def run_one(x_oh_seq):
            logits, _ = model.apply(params, x_oh_seq, h0)  # (T,V)
            return logits
        return jax.vmap(run_one, in_axes=0)(x_ids)  # (B,T,V)

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
        return loss
    
    return optimizer, train_step, eval_step

def generate(model, params, dm, prompt: str, max_new_tokens: int = 100):
    ids = dm.encode_str(prompt)
    context_ids = list(ids)
    h0 = jnp.zeros((HIDDEN_DIM,), dtype=jnp.float32)

    for _ in range(max_new_tokens):
        x_ids = jnp.array(context_ids, dtype=jnp.int32) 
        logits, _ = model.apply(params, x_ids, h0)
        last_logit = logits[-1]
        next_id = int(jnp.argmax(last_logit))
        context_ids.append(next_id)
    
    return dm.decode_ids(context_ids)

def log_jsonl(path, **kv):
    with open(path, "a") as f:
        f.write(json.dumps(kv) + "\n")


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
    model, params = setup_model(V, rng, HIDDEN_DIM, MLP_HIDDEN, SEQ_LEN, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_LEN)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print("Total params:", param_count)

    optimizer, train_step, eval_step = make_train_step(model, V, HIDDEN_DIM, LR)
    opt_state = optimizer.init(params)

    es = EarlyStopper(patience=10, min_delta=1e-4)
    best_params = None 

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

            val_loss = eval_step(params, x_val, y_val)
            val_bpc = val_loss / jnp.log(2.0)
            val_ppl = jnp.exp(val_loss) 

            dt = time.time() - t0
            sec_per_step = dt / LOG_EVERY
            steps_per_sec = LOG_EVERY / dt
            chars_per_sec = LOG_EVERY * BATCH_SIZE * SEQ_LEN / dt
            print(f"[step {step:5d}] | train_loss = {float(loss):.4f} | val_loss = {float(val_loss):.4f} | "
                f"ppl={val_ppl:.3f} | bpc={val_bpc:.3f} | sec/step={sec_per_step:.4f} | steps/sec={steps_per_sec:.2f} | "
                f"chars/sec={chars_per_sec:.2f} ({dt:.1f}s)"
            )

            metrics = {
                "event": "log",
                "step": step,
                "train_loss": float(loss),
                "val_loss": float(val_loss),
                "ppl": float(val_ppl),
                "bpc": float(val_bpc),
                "sec_per_step": float(sec_per_step),
                "steps_per_sec": float(steps_per_sec),
                "chars_per_sec": float(chars_per_sec),
                "interval_sec": float(dt)
            }
            log_jsonl(JSONL_PATH, **metrics)

            should_stop, is_new_best = es.update(float(val_loss), step)
            if is_new_best:
                best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
                log_jsonl(JSONL_PATH, event="best_update", step=step, best_val=es.best_value)

            if should_stop:
                log_jsonl(JSONL_PATH, event="early_stop",
                        at_step=step, best_step=es.best_step,
                        best_val=es.best_value, patience=es.patience)
                break
            t0 = time.time()
    
    if best_params is not None:
        params = best_params
    log_jsonl(JSONL_PATH, event="run_end", best_step=es.best_step, best_val=es.best_value)
    
    try:
        prompt = "Litwo! Ojczyzny moja"
        sample = generate(model, params, dm, prompt=prompt, max_new_tokens=100)
        print(f"\n=== SAMPLE for {prompt} ===")
        print(sample)
    except Exception as e:
        print("Sampling failed:", e)


if __name__ == "__main__":
    main()
