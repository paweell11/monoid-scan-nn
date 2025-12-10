import optuna
import jax
import jax.numpy as jnp

from train import setup_model, make_train_step
from utils.early_stopping import EarlyStopper
from data import CharData

def train_one_run(hparams, trial=None) -> float:
    batch_size = hparams["batch_size"]
    seq_len = hparams["seq_len"]
    hidden_dim = hparams["hidden_dim"]
    mlp_hidden = hparams["mlp_hidden"]
    lr = hparams["lr"]
    max_steps = hparams["max_steps"]
    log_every = hparams["log_every"]
    patience = hparams.get("patience", 6)
    min_delta = hparams.get("min_delta", 1e-4)
    seed = hparams.get("seed", 0)

    d_model = hparams["d_model"]
    num_layers = hparams["num_layers"]
    num_heads = hparams["num_heads"]
    d_ff = hparams["d_ff"]
    max_len = hparams["max_len"]

    dm = CharData()
    dm.prepare()
    V = dm.vocab_size()

    train_iter = dm.train_loader(batch_size=batch_size, shuffle=True)
    val_iter   = dm.val_loader(batch_size=batch_size, shuffle=False)

    rng = jax.random.PRNGKey(seed)
    model, params = setup_model(V, rng, hidden_dim, mlp_hidden, seq_len, d_model, num_layers, num_heads, d_ff, max_len)
    optimizer, train_step, eval_step = make_train_step(model, V, hidden_dim, lr)
    opt_state = optimizer.init(params)

    es = EarlyStopper(patience=patience, min_delta=min_delta)
    best_params = None

    val_idx = 0

    for step in range(1, max_steps + 1):
        x_np, y_np = next(train_iter)
        x = jnp.array(x_np, jnp.int32)
        y = jnp.array(y_np, jnp.int32)
        params, opt_state, loss = train_step(params, opt_state, x, y)

        if step % log_every == 0:
            x_val_np, y_val_np = next(val_iter)
            x_val = jnp.array(x_val_np, jnp.int32)
            y_val = jnp.array(y_val_np, jnp.int32)
            val_loss = eval_step(params, x_val, y_val)

            if trial is not None:
                val_idx += 1
                trial.report(float(val_loss), val_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            should_stop, is_new_best = es.update(float(val_loss), step)
            if is_new_best:
                best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
            if should_stop:
                break

    if best_params is not None:
        params = best_params

    return float(es.best_value)


def objective(trial: optuna.Trial) -> float:
    hparams = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        "hidden_dim": 64,
        "mlp_hidden": 96,

        "batch_size": 16,
        "seq_len": 128,
        "max_steps": 6000,
        "log_every": 200,     
        "patience": 10,
        "min_delta": 1e-4,
        "seed": 1,

        "d_model": 24,
        "num_layers": 2,
        "num_heads": 2, 
        "d_ff": 96,
        "max_len": 128,
    }
    best_val = train_one_run(hparams, trial=trial)
    return best_val


if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)