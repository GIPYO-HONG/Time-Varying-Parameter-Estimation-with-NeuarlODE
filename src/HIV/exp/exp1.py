import jax.numpy as jnp
from HIV.data.data import make_data
from HIV.models import *
from HIV.exp.Experiment import Experiment
import optax

exp_name = "ANODE"

model = an

patid = 16 #1~45
ts, ys, y0, scale = make_data(patid)

ts_eval = jnp.linspace(ts[0], ts[-1], len(ts) * 4 - 1)

EX = Experiment(
    model_cls = model.Main,
    y0=y0,
    ts=ts,
    ys=ys,
    norm_scale=scale,
    exp_name=exp_name,
)

steps = 10000

def adamw(lr, wd=1e-3):
    optimizer = optax.adamw(
        learning_rate=lr,
        weight_decay=wd,
    )
    return optimizer

if __name__=="__main__":
    EX.train(optimizer=adamw, lr=1e-3, steps=steps)