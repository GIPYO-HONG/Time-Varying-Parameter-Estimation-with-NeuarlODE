import jax.numpy as jnp
from simple.data1 import data
import simple.model as model
from simple.exp import Experiment

exp_name = "sim1"

y0, ts, ys = data()

ts_eval = jnp.linspace(ts[0], ts[-1], len(ts) * 4 - 1)

EX = Experiment(
    model_cls = model.Main,
    y0=y0,
    ts=ts,
    ys=ys,
    exp_name=exp_name,
)

steps = 100000

if __name__=="__main__":
    EX.train(lr=1e-5, steps=steps)