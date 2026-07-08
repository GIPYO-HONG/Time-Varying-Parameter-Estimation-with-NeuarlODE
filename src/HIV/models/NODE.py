########## module import ##########
import jax
import jax.random as jr
import jax.numpy as jnp
import jax.nn as jnn

import equinox as eqx
import diffrax



########## model define ##########

class Eta(eqx.Module):
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
        eta_out = self.mlp(t_input)
        return eta_out.squeeze()
    
class Main(eqx.Module):
    eta: Eta
    norm_scale: tuple = eqx.field(static=True)

    y0: jnp.ndarray

    def __init__(self, y0, hidden_dim, width_size, depth, norm_scale, *, key):

        self.norm_scale = tuple(norm_scale.tolist())

        self.eta = Eta(width_size, depth, key=key)

        self.y0 = jnp.array([y0[0], 0, y0[1]])
    
    def RHS(self, t, y, args=None):
        norm_state = y

        scale = jnp.array(self.norm_scale)
        state = norm_state * scale
        Tu, Ti, V = state

        ee = self.eta(t)

        ll, d, dd, N, c = 10.0, 0.01, 0.7, 100, 13.0

        dTu = ll - d*Tu - ee*V*Tu
        dTi = ee*V*Tu - dd*Ti
        dV = N*dd*Ti - c*V

        dstate = jnp.array([dTu, dTi, dV])
        dnorm_state = dstate / scale

        return dnorm_state
    
    def loss(self, ts, ys):
        pred = self.__call__(ts)

        T_pred = pred[:, 0] + pred[:, 1]
        V_pred = pred[:, 2]

        T_target = ys[:, 0] / self.norm_scale[0]
        V_target = ys[:, 1] / self.norm_scale[2]

        loss = jnp.mean(jnp.square(T_pred - T_target)) \
             + jnp.mean(jnp.square(V_pred - V_target))
        return loss 
    
    def eval(self, ts_eval):
        ys_pred = self.__call__(ts_eval)
        ys_pred = ys_pred * jnp.array(self.norm_scale)

        eta_pred = jax.vmap(lambda t: self.eta(jnp.array([t])))(ts_eval)

        return ys_pred, eta_pred
    
    def __call__(self, ts):
        y0 = self.y0
        scale = jnp.array(self.norm_scale)
        norm_y0 = y0 / scale

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.RHS),
            # diffrax.Tsit5(),
            diffrax.Kvaerno5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=0.001,
            y0=norm_y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            max_steps=50000,
        )

        return sol.ys