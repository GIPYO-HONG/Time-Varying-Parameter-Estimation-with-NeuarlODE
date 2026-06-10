# # exp.py 상단에 임시 추가
# import os
# os.environ["JAX_DISABLE_JIT"] = "1"

import jax.numpy as jnp
import jax.random as jr
from models import *

exp_name = "exp2"

model = tmp

key = jr.PRNGKey(5678)
key1, key2 = jr.split(key)

y0 = jnp.array([600., 30., 10**5])
ts = jnp.linspace(0., 20., 30)
ys = get_data(ts, y0, eta)

T_clean = ys[:,0] + ys[:,1]
V_clean = ys[:,2]

T_data = T_clean + jr.normal(key1, shape=T_clean.shape) * jnp.sqrt(1600)
V_data = V_clean + jr.normal(key2, shape=V_clean.shape) * jnp.sqrt(10000)

ys = jnp.stack([T_data, V_data], axis=-1)

ts_eval = jnp.linspace(0., 20., 800)

EX = model.Experiment(
    ts=ts,
    ys=ys,
    eta=eta, 
    exp_name=exp_name,
)

steps = 5000

if __name__=="__main__":
    EX.train(lr=1e-3, steps=steps)