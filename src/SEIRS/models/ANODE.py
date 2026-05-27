######## module import ########
import jax
import jax.random as jr
import jax.numpy as jnp
import jax.nn as jnn

import equinox as eqx
import diffrax

from .utiles import *

######## model define ########

class Dynamics(eqx.Module):
    scale: jnp.ndarray
    mlp: eqx.nn.MLP

    def __init__(self, hidden_dim, width_size, depth, *, key):
        self.scale = jnp.array(0.1)
        self.mlp = eqx.nn.MLP(
            in_size=hidden_dim + 4,
            out_size=hidden_dim,
            width_size=width_size,
            depth=depth,
            activation=lambda x: jnn.softplus(x),
            final_activation=lambda x: jnn.tanh(0.0001*x),
            key=key,
        )

    def __call__(self, t, h, args=None):
        return self.scale * self.mlp(h)


class Argphy(eqx.Module):
    hidden_dyn: Dynamics

    hidden_vec: jnp.ndarray

    hidden_to_beta: eqx.nn.Linear
    hidden_to_sigma: eqx.nn.Linear

    y0: jnp.ndarray

    def __init__(self, hidden_dim, width_size, depth, *, key):
        beta_key, hvec_key, htb_key, hts_key = jr.split(key, 4)

        self.hidden_dyn = Dynamics(hidden_dim, width_size, depth, key=beta_key)

        self.hidden_vec = 0.01 * jr.normal(hvec_key, (hidden_dim,))

        self.hidden_to_beta = eqx.nn.Linear(hidden_dim, 1, key=htb_key)
        self.hidden_to_sigma = eqx.nn.Linear(hidden_dim, 1, key=hts_key)
        
        self.y0 = jnp.array([4865., 9., 68., 0.])

    def get_param(self, h):
        beta = jnn.softplus(self.hidden_to_beta(h))
        sigma = jnn.softplus(self.hidden_to_sigma(h))
        return beta.squeeze(), sigma.squeeze()

    def RHS(self, t, y, args=None):
        state, h = y

        S, E, I, R = state
        N = S+E+I+R

        mm, dd, r, kk, aa, gg = 0.0003671, 0.0027400, 0.0006762, 0.0001500, 0.0300000, 0.3500000

        bb, ss = self.get_param(h)

        dS = - bb * I*S / N - mm*S + r*N + dd*R
        dE = bb * I * S / N - (mm + ss + kk)*E
        dI = ss*E - (mm + aa + gg)* I
        dR = kk*E + gg*I - mm*R - dd*R

        dstate = jnp.array([dS, dE, dI, dR])

        dh = self.hidden_dyn(t, jnp.concatenate([h, state]), args)

        return (dstate, dh)

    def __call__(self, ts):
        y0 = jnn.softplus(self.y0)
        h0 = self.hidden_vec

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.RHS),
            diffrax.Tsit5(),
            # diffrax.Kvaerno5(),
            # diffrax.Dopri8(),
            t0=ts[0],
            t1=ts[-1],
            dt0=0.001,
            y0=(y0, h0),
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            max_steps=50000,
        )

        states, h = sol.ys

        return states, h

########## Experiment ##########

class Experiment(BaseExperiment):

    def __init__(self, ts, ys, hidden_dim=16, width_size=64, depth=4, **kwargs):

        seed = kwargs.get("seed", 5678)

        model = Argphy(
            hidden_dim,
            width_size,
            depth,
            key=jax.random.PRNGKey(seed),
        )

        super().__init__(model, ts, ys, **kwargs)

    def loss_fn(self, model, ts, ys):
        pred, _ = model(ts)
        loss = jnp.mean(jnp.square(pred[:,2] - ys) / jnp.max(ys).squeeze())
        return loss


########## Evaluation ##########

def Evaluation(EX, ts_eval, loss_list):
    ts_data, ys_data, model = EX.ts, EX.ys, EX.model

    ys_pred, h_pred = model(ts_eval)

    beta_pred, sigma_pred = jax.vmap(model.get_param)(h_pred)

    plotting(ts_data, ys_data, ts_eval, ys_pred, beta_pred, sigma_pred, loss_list)