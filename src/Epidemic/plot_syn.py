import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RUN_NAME = Path(__file__).stem

# Data

y0 = jnp.array([5e7, 20000., 40000., 720000., 0.])
POPULATION = jnp.sum(y0)
ts = jnp.linspace(0, 365, 366)

params = jnp.array([1/5, 1/10, 1/180]) #dd, gg, aa

def beta(t):
    period = 365 / 3
    decay = 0.6 ** (t / period)
    return 0.1 * decay * jnp.sin(2 * jnp.pi * t /period) + 0.15

def func(t, y, args):
    dd, gg, aa = args
    S, E, I, R, C = y

    dS = - beta(t) * S * I / POPULATION + aa * R
    dE = beta(t) * S * I / POPULATION - dd * E
    dI = dd * E - gg * I
    dR = gg * I - aa * R
    dC = dd * E
    return jnp.array([dS, dE, dI, dR, dC])

def solve_ivp(y0, ts):
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(func),
        diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1]-ts[0],
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        args=params,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8)
    )

    return sol.ys

def make_data(y0, ts):
    state = solve_ivp(y0, ts)
    C = state[:, 4]
    ys = jnp.diff(C, prepend=C[:1])

    return ts, ys

if __name__ == "__main__":
    ts, ys = make_data(y0, ts)

    fig, axes = plt.subplots(2, 1 ,figsize=(14, 7))

    axes[0].scatter(ts, ys, label="data", s=10, color="tab:blue")
    axes[0].legend()
    axes[0].set_xlabel("days")
    axes[0].set_ylabel("incidence")

    axes[1].plot(ts, beta(ts), label="beta(t)", color="tab:orange")
    axes[1].legend()
    axes[1].set_xlabel("days")

    fig.tight_layout()
    fig.savefig(BASE_DIR / "synthetic_data.png", dpi=200, bbox_inches="tight")
    plt.close(fig)