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
path = Path(__file__).with_name("COVID19_total_cases.csv")

df = pd.read_csv(path, encoding="utf-8-sig")

df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
df = df.sort_values("date").reset_index(drop=True)

y0 = jnp.array([5e7, 20000., 40000., 720000.])
POPULATION = jnp.sum(y0)
params = jnp.array([1 / 3.5, 1 / 5])

def make_data(start=None, end=None):
    """
    설명
    """
    sub = df.copy()

    if start is not None:
        sub = sub[sub["date"] >= pd.to_datetime(start)]

    if end is not None:
        sub = sub[sub["date"] <= pd.to_datetime(end)]

    sub = sub.reset_index(drop=True)

    ts = jnp.array((sub["date"] - sub["date"].iloc[0]).dt.days.values)

    ys = jnp.array(sub["cases"].values)

    # normalization

    days = ts[-1] - ts[0]

    y0_normal = y0 / POPULATION    
    ts_normal = ts / days
    ys_normal = ys / POPULATION 

    return y0_normal, ts_normal, ys_normal, days

# Model

def softplus_inverse(value):
    """Map a positive physical value to its unconstrained softplus parameter."""
    return value + jnp.log(-jnp.expm1(-value))

class Beta(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, width_size, depth, *, key):

        self.mlp = eqx.nn.MLP(
            in_size = 1,
            out_size=1,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            final_activation=jnn.softplus,
            key=key,
        )

    def __call__(self, t):
        return self.mlp(jnp.atleast_1d(t)).squeeze()

# NUM = 5

# def softplus_inverse(value):
#     """Map a positive physical value to its unconstrained softplus parameter."""
#     return value + jnp.log(-jnp.expm1(-value))

# class Beta(eqx.Module):
#     mlp: eqx.nn.MLP
#     num: int = eqx.field(static=True)

#     def __init__(self, width_size, depth, *, key, num=NUM):
#         self.num = num

#         self.mlp = eqx.nn.MLP(
#             in_size = 2 * self.num + 1,
#             out_size=1,
#             width_size=width_size,
#             depth=depth,
#             activation=jnn.tanh,
#             final_activation=jnn.softplus,
#             key=key,
#         )

#     def __call__(self, t):
#         freq = 2.0 ** jnp.arange(self.num)

#         angles = 2.0 * jnp.pi * freq * t

#         features = jnp.concatenate([
#             jnp.atleast_1d(t),
#             jnp.sin(angles),
#             jnp.cos(angles),
#         ])

#         out = self.mlp(features)

#         return out.squeeze()

class Main(eqx.Module):
    beta: Beta

    raw_y0: jnp.ndarray

    def __init__(self, width_size, depth, y0, *, key):
        self.beta = Beta(width_size, depth, key=key)

        self.raw_y0 = softplus_inverse(y0)

    @property
    def y0(self):
        y0 = jnn.softplus(self.raw_y0)
        return jnp.concatenate([y0, jnp.asarray([0], dtype=y0.dtype)])

    def func(self, t, y, args):
        S, E, I, R, C = y

        bb = self.beta(t)

        params, d = args
        dd, gg = params

        dS = - d * bb * S * I
        dE = - d * dd * E + d * bb * S * I
        dI = d * dd * E - d * gg * I
        dR = d * gg * I
        dC = d * dd * E

        dy = jnp.array([dS, dE, dI, dR, dC])

        return dy

    def __call__(self, ts, days):
        y0 = self.y0

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1]-ts[0],
            y0=y0,
            args=(params, days),
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        )

        return sol.ys

# Training

class Experiment:
    def __init__(self, y0, ts, ys, days, width_size=64, depth=8, seed=5678):
        self.model = Main(width_size, depth, y0, key=jr.PRNGKey(seed))
        self.ts, self.ys, self.days = ts, ys, days

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
            pred_ = model(self.ts, self.days)[:,4]
            pred = jnp.diff(pred_)
            observe = self.ys[1:]
            scale = jnp.maximum(jnp.max(observe), jnp.finfo(self.ys.dtype).eps)
            data_loss = jnp.mean(jnp.square((pred-observe) / scale))

            if lam_d2 == 0:
                return data_loss

            dt = jnp.mean(jnp.diff(self.ts))
            d2_beta = jnp.diff(jax.vmap(model.beta)(self.ts), n=2, axis=0) / dt ** 2
            smooth_loss = jnp.mean(jnp.square(d2_beta))
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

    ts_normal_data, ys_normal_data, days = exp.ts, exp.ys, exp.days
    ts_data, ys_data = ts_normal_data * days, ys_normal_data * POPULATION
    
    ts_normal_eval = jnp.linspace(ts_normal_data[0], ts_normal_data[-1], (len(ts_normal_data) - 1) * 4 + 1)
    ts_eval = ts_normal_eval * days

    pred_normal_eval = exp.model(ts_normal_eval, days)
    pred_eval = pred_normal_eval * POPULATION

    pred_data = exp.model(ts_normal_data, days) * POPULATION
    incidence_pred = jnp.diff(pred_data[:, 4])
    ts_incidence = ts_data[1:]
    ys_incidence = ys_data[1:]
    beta_eval = jax.vmap(exp.model.beta)(ts_normal_eval)

    best_loss = exp.best_loss

    # data + pred // beta
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    axes[0].plot(ts_incidence, incidence_pred, label="pred", color="tab:blue")
    axes[0].scatter(ts_incidence, ys_incidence, label="data", s=4, color="tab:orange")
    axes[0].set_title("Observed Data and Model Prediction")
    axes[0].legend()

    axes[1].plot(ts_eval, beta_eval, label="beta", color="tab:green")
    axes[1].set_title("Estimated Parameter")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(results_dir / "data_params.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # state
    STATE_NAMES = ("S", "E", "I", "R")
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    for index, (ax, name) in enumerate(zip(axes, STATE_NAMES)):
        ax.plot(ts_eval, pred_eval[:, index], color="tab:orange", label="Prediction")
        ax.set_ylabel(f"{name} (people)")
        ax.legend()
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (days)")
    fig.suptitle("Predicted States", fontsize=16)
    fig.tight_layout()
    fig.savefig(results_dir / "states.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # loss / 확인용
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.semilogy(np.arange(1, losses.size + 1), losses, label="loss", color="tab:blue")
    ax.legend()
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(results_dir / "loss.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # residual
    residual = ys_incidence - incidence_pred
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axhline(0, color="tab:gray", linestyle="--", linewidth=1)
    ax.plot(ts_incidence, residual, ".", color="tab:red")
    ax.set_xlabel("Days since first observation")
    ax.set_ylabel("Observed − predicted cases")
    ax.set_title("Observation residuals")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(results_dir / "residual.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Results
    lines = [
        f"experiment: {RUN_NAME}",
        f"best checkpoint loss: {best_loss:.8e}",
        f"incidence l2 relative error: {l2_rel_error(incidence_pred, ys_incidence):.8e}",
    ]

    learned_y0 = np.asarray(exp.model.y0) * POPULATION
    lines.append("initial states (people):")
    for name, value in zip(("S0", "E0", "I0", "R0"), learned_y0[:4]):
        lines.append(f"{name}: {value:.8e}")

    (results_dir / "error.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    # training part
    y0, ts, ys, days = make_data("2022-01", "2023-04")
    exp = Experiment(y0, ts, ys, days)
    exp.train(lr=1e-5, steps=50000, lam_d2 = 0.)

    # # evaluation part
    evaluate(exp)