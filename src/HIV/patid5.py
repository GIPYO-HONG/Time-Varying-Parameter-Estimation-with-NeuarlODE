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
import pandas as pd
import time

BASE_DIR = Path(__file__).resolve().parent
RUN_NAME = Path(__file__).stem

# Data

path = BASE_DIR / "ACTG315.csv"
df = pd.read_csv(path, sep=r"\s+", header=0, names=["obs_no", "patid", "day", "log10_rna", "cd4"])
df = df.astype({"patid": int, "day": float, "log10_rna": float, "cd4": float})
df = df.sort_values(["patid", "day"]).reset_index(drop=True)

params = jnp.array([19.9, 0.01, 0.7, 2000, 13]) # ll, d, dd, N, c

def make_data(patid):
    sub = df[df["patid"] == patid]

    ts = jnp.array(sub["day"])

    T = jnp.array(sub["cd4"])
    V = 10 ** jnp.array(sub["log10_rna"])
    ys = jnp.stack([T, V], axis=1)

    y0 = ys[0,:]

    # normalization

    days = ts[-1] - ts[0]
    scale = jnp.max(ys, axis=0) # [T_max, V_max]

    y0_normal = y0 / scale
    ts_normal = ts / days
    ys_normal = ys / scale
   
    return y0_normal, ts_normal, ys_normal, days, scale

# Model

def softplus_inverse(value):
    """Map a positive physical value to its unconstrained softplus parameter."""
    return value + jnp.log(-jnp.expm1(-value))

# NUM = 5

# class Param(eqx.Module):
#     mlp: eqx.nn.MLP

#     def __init__(self, width_size, depth, *, key):

#         self.mlp = eqx.nn.MLP(
#             in_size = 2 * NUM + 1,
#             out_size=1,
#             width_size=width_size,
#             depth=depth,
#             activation=jnn.tanh,
#             final_activation=jnn.softplus,
#             key=key,
#         )

#     def __call__(self, t):
#         freq = 2.0 ** jnp.arange(NUM)

#         angles = 2.0 * jnp.pi * freq * t

#         features = jnp.concatenate([
#             jnp.atleast_1d(t),
#             jnp.sin(angles),
#             jnp.cos(angles),
#         ])

#         out = self.mlp(features)

#         return out.squeeze()

class Param(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, width_size, depth, *, key):

        self.mlp = eqx.nn.MLP(
            in_size = 1,
            out_size=1,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            final_activation=lambda x: 0.001 * jnn.sigmoid(x),
            key=key,
        )

    def __call__(self, t):
        return self.mlp(jnp.atleast_1d(t)).squeeze()

class Main(eqx.Module):
    k: Param
    raw_y0: jnp.ndarray
    log_params: jnp.ndarray

    def __init__(self, width_size, depth, y0, scale, *, key):
        self.k = Param(width_size, depth, key=key)
        self.log_params = jnp.log(params)

        I0 = jnp.minimum(20 / scale[0], y0[0] / 2)
        rraw_y0 = jnp.array([y0[0] - I0, I0, y0[1]])
        self.raw_y0 = softplus_inverse(jnp.maximum(rraw_y0, 1e-6))

    @property
    def y0(self):
        states = jnn.softplus(self.raw_y0)
        return states


    @property
    def params(self):
        # Learn relative changes while keeping all physical parameters positive.
        return jnp.exp(self.log_params)

    def func(self, t, y, args):
        S, I, V = y

        k = self.k(t)

        params, days, scale = args
        ll, d, dd, N, c = params
        ss1, ss2 = scale

        dS = days * ll / ss1 - days * d * S - days * k * ss2 * V * S
        dI = days * k * ss2 * V * S - days * dd * I
        dV = N * days * dd * (ss1 / ss2) * I - days * c * V

        dy = jnp.array([dS, dI, dV])

        return dy

    def __call__(self, ts, days, scale):

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            # diffrax.Kvaerno5(),
            t0=ts[0],
            t1=ts[-1],
            # dt0=ts[1]-ts[0],
            dt0=1e-5,
            y0=self.y0,
            args=(self.params, days, scale),
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            max_steps=50000,
        )

        return sol.ys

# Training

class Experiment:
    def __init__(self, y0, ts, ys, days, scale, width_size=64, depth=8, seed=5678):
        self.model = Main(width_size, depth, y0, scale, key=jr.PRNGKey(seed))
        self.ts, self.ys, self.days, self.scale = ts, ys, days, scale

        self.weights_dir = Path(BASE_DIR) / "weights" / RUN_NAME

        self.ckpt_dir = self.weights_dir / "model_weights"
        self.loss_path = self.weights_dir / "loss_list.npy"
        self.best_weight_path = self.weights_dir / "best_weight.eqx"
        
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.loss_list = []
        self.best_loss = float("inf")

    def save_ckpt(self, step):
        eqx.tree_serialise_leaves(self.ckpt_dir / f"model_step_{step:05d}.eqx", self.model)

    def train(self, lr=1e-3, steps=10000, viz_loss=1000):
        if steps <= 0 or viz_loss <= 0:
            raise ValueError("steps and viz_loss must be positive")

        params, static = eqx.partition(self.model, eqx.is_inexact_array)

        optim= optax.adamw(lr)
        opt_state = optim.init(params)

        def loss_fn(params):
            model = eqx.combine(params, static)
            pred = model(self.ts, self.days, self.scale)

            T_pred = pred[:, 0] + pred[:, 1]
            V_pred = pred[:, 2]
            ys_pred = jnp.stack([T_pred, V_pred], axis=1)

            data_loss = jnp.mean(jnp.square(ys_pred - self.ys))

            return data_loss
            
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

    ts_normal_data, ys_normal_data, days, scale = exp.ts, exp.ys, exp.days, exp.scale
    ts_data, ys_data = ts_normal_data * days, ys_normal_data * scale

    ts_normal_eval = jnp.linspace(ts_normal_data[0], ts_normal_data[-1], 1000)
    ts_eval = ts_normal_eval * days

    state_scale = jnp.array([scale[0], scale[0], scale[1]])
    pred_eval = exp.model(ts_normal_eval, days, scale) * state_scale
    pred_data = exp.model(ts_normal_data, days, scale) * state_scale
    T_eval = pred_eval[:, 0] + pred_eval[:, 1]
    V_eval = pred_eval[:, 2]
    observed_pred = jnp.stack([pred_data[:, 0] + pred_data[:, 1], pred_data[:, 2]], axis=1)
    k_eval = jax.vmap(exp.model.k)(ts_normal_eval)

    # data + pred // k
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for index, (name, prediction) in enumerate(zip(("T", "V"), (T_eval, V_eval))):
        axes[index].plot(ts_eval, prediction, label="Prediction", color="tab:blue")
        axes[index].scatter(ts_data, ys_data[:, index], label="Data", s=20, color="tab:orange")
        axes[index].set_ylabel(name)
        axes[index].set_title(f"Observed and Predicted {name}")
        axes[index].legend()
    axes[2].plot(ts_eval, k_eval, label="k", color="tab:green")
    axes[2].set_ylabel("k")
    axes[2].set_title("Estimated k(t)")
    axes[2].legend()
    axes[-1].set_xlabel("Time (days)")
    fig.tight_layout()
    fig.savefig(results_dir / "data_params.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # state: S, I, T (T = S + I)
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for ax, name, description, prediction in zip(
        axes, ("S", "I", "T"),
        ("Uninfected CD4 T cells", "Infected CD4 T cells", "Total CD4 T cells (S + I)"),
        (pred_eval[:, 0], pred_eval[:, 1], T_eval),
    ):
        ax.plot(ts_eval, prediction, color="tab:blue", label=f"{name}: {description} (prediction)")
        ax.set_ylabel(f"{name}: {description}")
        ax.grid(alpha=0.25)
    axes[2].scatter(ts_data, ys_data[:, 0], color="tab:orange", s=20, label="Observed total CD4 T cells")
    for ax in axes:
        ax.legend()
    axes[-1].set_xlabel("Time (days)")
    fig.suptitle("Predicted States", fontsize=16)
    fig.tight_layout()
    fig.savefig(results_dir / "states.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # loss
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.semilogy(np.arange(1, losses.size + 1), losses, label="loss", color="tab:blue")
    ax.legend()
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(results_dir / "loss.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # residual: observed - predicted at observation times
    residual = ys_data - observed_pred
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for index, (ax, name) in enumerate(zip(axes, ("T", "V"))):
        ax.axhline(0, color="tab:gray", linestyle="--", linewidth=1)
        ax.plot(ts_data, residual[:, index], ".", color="tab:red", label=f"{name} residual")
        ax.set_ylabel(f"Observed − predicted {name}")
        ax.set_title(f"{name} residuals")
        ax.legend()
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (days)")
    fig.tight_layout()
    fig.savefig(results_dir / "residual.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Results
    lines = [
        f"experiment: {RUN_NAME}",
        f"best checkpoint loss: {exp.best_loss:.8e}",
        f"T l2 relative error: {l2_rel_error(observed_pred[:, 0], ys_data[:, 0]):.8e}",
        f"V l2 relative error: {l2_rel_error(observed_pred[:, 1], ys_data[:, 1]):.8e}",
    ]
    learned_y0 = np.asarray(exp.model.y0 * state_scale)
    lines.append("initial states (original data units):")
    for name, value in zip(("S0", "I0", "V0"), learned_y0):
        lines.append(f"{name}: {value:.8e}")
    lines.append(f"T0: {learned_y0[0] + learned_y0[1]:.8e}")
    lines.append("learned parameters:")
    for name, value in zip(("lambda", "d", "delta", "N", "c"), np.asarray(exp.model.params)):
        lines.append(f"{name}: {value:.8e}")
    (results_dir / "error.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    # training part
    patid = 5 #1~45
    y0, ts, ys, days, scale = make_data(patid)
    exp = Experiment(y0, ts, ys, days, scale)
    exp.train(lr=1e-5, steps=500000, viz_loss=1000)

    # # evaluation part
    evaluate(exp)