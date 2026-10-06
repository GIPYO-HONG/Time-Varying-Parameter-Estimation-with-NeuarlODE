import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
import numpy as np

import diffrax
import equinox as eqx
import optax

import matplotlib.pyplot as plt
from pathlib import Path
import time
import os

BASE_DIR = Path(__file__).resolve().parent
RUN_NAME = Path(__file__).stem

# Data

#=====================================================================================
# example 1 setting
y0 = jnp.array([0.02])
ts = jnp.linspace(0, 1, 100)

def func(t, y, args=None):
    return 8.0 * (1-y) * y
#=====================================================================================

def solve_ivp(y0, ts):
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(func),
        diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1]-ts[0],
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
    )

    return y0, sol.ts, sol.ys

# Model

class Param(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, width_size, depth, *, key):

        self.mlp = eqx.nn.MLP(
            in_size = 1,
            out_size=1,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            key=key,
        )

    def __call__(self, t):
        return self.mlp(jnp.atleast_1d(t)).squeeze()

class Main(eqx.Module):
    param: Param

    def __init__(self, width_size, depth, *, key):
        self.param = Param(width_size, depth, key=key)

    def RHS(self, t, y, args=None):
        return self.param(t) * y

    def __call__(self, y0, ts):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.RHS),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1]-ts[0],
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        )

        return sol.ys

# Training

class Experiment:
    def __init__(self, y0, ts, ys, width_size=64, depth=2, seed=5678):
        self.model = Main(width_size, depth, key=jr.PRNGKey(seed))
        self.y0, self.ts, self.ys = y0, ts, ys

        self.weights_dir = Path(BASE_DIR) / "weights" / RUN_NAME

        self.ckpt_dir = self.weights_dir / "model_weights"
        self.loss_path = self.weights_dir / "loss_list.npy"
        self.best_weight_path = self.weights_dir / "best_weight.eqx"
        
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.loss_list = []
        self.best_loss = float("inf")

    def save_ckpt(self, step):
        eqx.tree_serialise_leaves(self.ckpt_dir / f"model_step_{step:05d}.eqx", self.model)

    def train(self, lr=1e-3, steps=10000, viz_loss=1000, lam_d2=0.):
        if steps <= 0 or viz_loss <= 0:
            raise ValueError("steps and viz_loss must be positive")

        params, static = eqx.partition(self.model, eqx.is_inexact_array)

        optim= optax.adamw(lr)
        opt_state = optim.init(params)

        def loss_fn(params):
            model = eqx.combine(params, static)
            pred = model(self.y0, self.ts)
            data_loss = jnp.mean(jnp.square(pred-self.ys))

            if lam_d2 == 0:
                return data_loss

            dt = jnp.mean(jnp.diff(self.ts))
            d2_param = jnp.diff(jax.vmap(model.param)(self.ts), n=2, axis=0) / dt ** 2
            smooth_loss = jnp.mean(jnp.square(d2_param))
            return data_loss + lam_d2 * smooth_loss

        loss_and_grad = eqx.filter_value_and_grad(loss_fn)

        def keep_best(new, old, better):
            return jax.tree.map(lambda x, y: jnp.where(better, x, y), new, old)

        def step(carry, _):
            params, opt_state, best, best_loss = carry
            loss, grads = loss_and_grad(params)
            better = loss < best_loss
            updates, opt_state = optim.update(grads, opt_state, params)
            return (eqx.apply_updates(params, updates), opt_state, keep_best(params, best, better), jnp.where(better, loss, best_loss)), loss

        @eqx.filter_jit
        def train_steps(params, opt_state, best, best_loss, count):
            return jax.lax.scan(step, (params, opt_state, best, best_loss), None, length=count)

        best, best_loss = params, jnp.array(jnp.inf)


        print(f"Training Start: {RUN_NAME}")

        for epoch in range(0, steps, viz_loss):
            count = min(viz_loss, steps - epoch)
            t0 = time.time()
            (params, opt_state, best, best_loss), losses = train_steps(params, opt_state, best, best_loss, count)

            self.loss_list.extend(np.asarray(losses).tolist())
            self.model = eqx.combine(params, static)
            self.save_ckpt(epoch + count)

            print(f"step: {epoch + count:5d}, loss: {losses[-1]:.6e}, time: {time.time() - t0:.2f}s")

    
        final_loss = loss_fn(params)
        better = final_loss < best_loss
        self.model = eqx.combine(keep_best(params, best, better), static)
        self.best_loss = float(jnp.where(better, final_loss, best_loss))
        print(f"Finish Training, best loss: {self.best_loss:.6e}")

        eqx.tree_serialise_leaves(self.best_weight_path, self.model)
        np.save(self.loss_path, np.asarray(self.loss_list))
        print("Save best weight and loss list")

# Evaluation

def l2_rel_error(pred, true):
    true = np.asarray(true, dtype=float)
    error = np.linalg.norm(np.asarray(pred, dtype=float) - true)
    norm = np.linalg.norm(true)
    return error / norm if norm else (0.0 if error == 0 else float("inf"))

def evaluate(exp):
    results_dir = Path(BASE_DIR) / "results" / RUN_NAME
    results_dir.mkdir(parents=True, exist_ok=True)

    exp.model = eqx.tree_deserialise_leaves(exp.best_weight_path, exp.model)
    losses = np.load(exp.loss_path)

    ts_data, y0 = exp.ts, exp.y0
    ts_eval = jnp.linspace(ts_data[0], ts_data[-1], (len(ts_data) - 1) * 4 + 1)

    pred_eval = exp.model(y0, ts_eval)
    true_eval = solve_ivp(y0, ts_eval)[2]
    param_eval = jax.vmap(exp.model.param)(ts_eval)

    best_loss = exp.best_loss

    # data + pred // param
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].plot(ts_eval, pred_eval, label="pred", color="tab:blue", zorder=1)
    axes[0].scatter(ts_data, exp.ys, label="data", s=5, color="tab:red", zorder=2)
    axes[0].set_title("Observed Data and Model Prediction")
    axes[0].legend()

    axes[1].plot(ts_eval, param_eval, label="param", color="tab:green")
    axes[1].set_title("Estimated Parameter")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(results_dir / "state_params.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # loss / 확인용
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.semilogy(np.arange(1, losses.size + 1), losses)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(results_dir / "loss.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Results
    lines = [
        f"experiment: {RUN_NAME}",
        f"best checkpoint loss: {best_loss:.8e}",
    ]

    lines.append(f"l2 relative error: {l2_rel_error(pred_eval, true_eval):.8e}")
    
    (results_dir / "error.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    # training part
    y0, ts, ys = solve_ivp(y0, ts)
    exp = Experiment(y0, ts, ys)
    # exp.train(lr=1e-5, steps=100000)

    # # evaluation part
    evaluate(exp)