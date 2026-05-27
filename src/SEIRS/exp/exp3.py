import jax.numpy as jnp
import pandas as pd
import os
from models import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'influenza_sydney_1919.csv')

df = pd.read_csv(csv_path)
ys = jnp.array(df['Total_Cases'].values, dtype=jnp.float32)

exp_name = "exp3"

model = tmp

ts_data = jnp.linspace(0., len(ys) - 1, len(ys))
ts_ge = jnp.linspace(0., len(ys) - 1, len(ys)*2)

ts_eval = jnp.linspace(0., len(ys) - 1, len(ys) * 4)

EX = model.Experiment(
    ts_data=ts_data,
    ys=ys,
    ts_ge=ts_ge,
    exp_name=exp_name,
)

steps = 100000

if __name__=="__main__":
    EX.train(lr=1e-3, steps=steps)