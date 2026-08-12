import jax
import jax.numpy as jnp
import jax.nn as jnn

import equinox as eqx
import diffrax

class Param(eqx.Module):
    mlp: eqx.nn.MLP
    dim: int = eqx.field(static=True)
    num: int = eqx.field(static=True)

    def __init__(self, dim, num, width_size, depth, *, key):
        self.dim = dim
        self.num = num

        self.mlp = eqx.nn.MLP(
            in_size=2 * self.num,
            out_size=self.dim ** 2,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            key=key,
        )

    def __call__(self, t):
        frequencies = 2.0 ** jnp.arange(self.num)

        angles = 2.0 * jnp.pi * frequencies * t

        features = jnp.concatenate([
            jnp.sin(angles),
            jnp.cos(angles),
        ])

        out = self.mlp(features)

        return out.reshape(self.dim, self.dim)

class Main(eqx.Module):
    param: Param

    def __init__(self, dim, num, width_size, depth, *, key):
        self.param = Param(dim, num, width_size, depth, key=key)

    def RHS(self, t, y, args=None):
        A = self.param(t)
        return A @ y

    def loss(self, y0, ts, ys):
        pred = self.__call__(y0, ts)
        data_loss = jnp.mean(jnp.square(pred - ys))

        dbeta_dt = jax.vmap(
            jax.grad(lambda t: self.param(t).squeeze())
        )(ts)
        smoothing_loss = jnp.mean(jnp.square(dbeta_dt))
        
        loss = data_loss + 1e-8 * smoothing_loss
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