########## module import ##########
import jax
import jax.random as jr
import jax.numpy as jnp
import jax.nn as jnn

import equinox as eqx
import diffrax


########## model define ##########

class Beta(eqx.Module):
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
        t_input = jnp.atleast_1d(t) 
        beta_out = self.mlp(t_input)
        return beta_out.squeeze()
    
class Main(eqx.Module):
    beta: Beta

    y0: jnp.ndarray

    def __init__(self, hidden_dim, width_size, depth, *, key):

        self.beta = Beta(width_size, depth, key=key)

        self.y0 = jnp.array([4865., 9., 68., 0.])

    def RHS(self, t, y, args=None):

        S, E, I, R = y
        N = S+E+I+R

        bb = self.beta(t)

        ss, mm, dd, r, kk, aa, gg = 0.850000, 0.0003671, 0.0027400, 0.0006762, 0.0001500, 0.0300000, 0.3500000

        dS = - bb * I*S / N - mm*S + r*N + dd*R
        dE = bb * I * S / N - (mm + ss + kk)*E
        dI = ss*E - (mm + aa + gg)* I
        dR = kk*E + gg*I - mm*R - dd*R

        dstate = jnp.array([dS, dE, dI, dR])

        return dstate
    
    def loss(self, ts, ys):
        pred = self.__call__(ts)
        loss = jnp.mean(jnp.square(pred[:,2] - ys) / jnp.max(ys).squeeze())
        return loss
    
    def eval(self, ts_eval):
        ys_pred = self.__call__(ts_eval)

        beta_pred = jax.vmap(lambda t: self.beta(jnp.array([t])))(ts_eval)

        return ys_pred, beta_pred
    
    def __call__(self, ts):
        y0 = jnn.softplus(self.y0)

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

