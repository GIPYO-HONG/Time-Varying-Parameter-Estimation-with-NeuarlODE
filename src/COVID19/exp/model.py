########## module import ##########
import jax
import jax.numpy as jnp
import jax.nn as jnn

import equinox as eqx
import diffrax

class Beta(eqx.Module):
    mlp: eqx.nn.MLP
    num: int = eqx.field(static=True)

    def __init__(self, width_size, depth, *, key):
        self.num = 5

        self.mlp = eqx.nn.MLP(
            in_size=2 * self.num,
            out_size=1,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            final_activation=jnn.sigmoid,
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

        return out.squeeze()
    
class Main(eqx.Module):
    beta: Beta

    # y0: jnp.ndarray
    y0: tuple = eqx.field(static=True)

    args: tuple = eqx.field(static=True)

    def __init__(self, width_size, depth, day, *, key):

        self.beta = Beta(width_size, depth, key=key)

        # self.y0 = jnp.array([1., 5./5e+7, 1./5e+7, 0./5e+7])
        self.y0 = (1., 5./5e+7, 1./5e+7, 0./5e+7)

        self.args = (1/5, 1/10, int(day))

    def RHS(self, t, y, args):
    
        S, E, I, R = y

        bb = self.beta(t)

        dd, gg, day = args

        dS = - day * bb* S * I
        dE = - day * dd * E + day * bb * S * I
        dI = day * dd * E - day * gg* I
        dR = day * gg * I

        dstate = jnp.array([dS, dE, dI, dR])

        return dstate
    
    def loss(self, ts, ys):
        pred = self.__call__(ts)
        scale = jnp.max(ys, axis=0)
        data_loss = jnp.mean(jnp.square((pred[:,2] - ys) / scale))
        return data_loss
    
    def eval(self, ts_eval):
        ys_pred = self.__call__(ts_eval)

        beta_pred = jax.vmap(lambda t: self.beta(jnp.array([t])))(ts_eval)

        return ys_pred, beta_pred
    
    def __call__(self, ts):
        # y0 = self.y0
        y0 = jnp.array(self.y0)

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.RHS),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=0.001,
            y0=y0,
            args=self.args,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-8),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            max_steps=50000,
        )

        return sol.ys


