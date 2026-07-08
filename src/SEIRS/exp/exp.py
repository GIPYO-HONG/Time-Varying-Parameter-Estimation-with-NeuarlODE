import jax.numpy as jnp
import pandas as pd
import os
from SEIRS.models import *
from SEIRS.exp.Experiment import Experiment

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "data", "influenza_sydney_1919.csv")

df = pd.read_csv(csv_path)
ys = jnp.array(df['Total_Cases'].values, dtype=jnp.float32)

exp_name = "exp"

model = no

ts = jnp.linspace(0., len(ys) - 1, len(ys))

ts_eval = jnp.linspace(0., len(ys) - 1, len(ys) * 4)

EX = Experiment(
    model_cls = model.Main,
    ts=ts,
    ys=ys,
    exp_name=exp_name,
)

steps = 10000

if __name__=="__main__":
    EX.train(lr=1e-4, steps=steps)
