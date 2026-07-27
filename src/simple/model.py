import jax
import jax.numpy as jnp
import jax.nn as jnn

import equinox as eqx
import diffrax

class Param(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, width_size, depth, *, key):
        self.mlp = eqx.nn.MLP(
            in_size=1,
            out_size=1,
            width_size=width_size,
            depth=depth,
            activation=lambda x: jnn.softplus(x),
            final_activation=lambda x: jnn.softplus(x),
            key=key
            )
        
    def __call__(self, t):
        out = self.mlp(jnp.atleast_1d(t))
        return out.squeeze()

class Main(eqx.Module):
    param: Param

    def __init__(self, width_size, depth, *, key):
        self.param = Param(width_size, depth, key=key)

    def RHS(self, t, y, args=None):
        pp = self.param(t)

        return pp * y

    def loss(self, y0, ts, ys):
        pred = self.__call__(y0, ts)

        loss = jnp.mean(jnp.square(pred - ys))

        return loss

    def eval(self, y0, ts_eval):
        ys_pred = self.__call__(y0, ts_eval)
        pp_pred = jax.vmap(lambda t: self.param(jnp.array([t])))(ts_eval)

        return ys_pred, pp_pred

    def __call__(self, y0, ts):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.RHS),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=0.001,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            max_steps=50000,
        )

        return sol.ys