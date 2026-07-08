######## module import ########
import jax
import jax.random as jr
import jax.numpy as jnp
import jax.nn as jnn

import equinox as eqx
import diffrax

######## model define ########

class Dynamics(eqx.Module):
    scale: jnp.ndarray
    mlp: eqx.nn.MLP

    def __init__(self, hidden_dim, width_size, depth, *, key):
        self.scale = jnp.array(0.1)
        self.mlp = eqx.nn.MLP(
            in_size=hidden_dim + 3,
            out_size=hidden_dim,
            width_size=width_size,
            depth=depth,
            activation=lambda x: jnn.softplus(x),
            final_activation=lambda x: jnn.tanh(0.0001*x),
            key=key,
        )

    def __call__(self, t, h, args=None):
        return self.scale * self.mlp(h)


class Main(eqx.Module):
    hidden_dyn: Dynamics
    hidden_vec: jnp.ndarray
    hidden_to_eta: eqx.nn.Linear
    norm_scale: tuple = eqx.field(static=True)

    y0: jnp.ndarray

    def __init__(self, y0, hidden_dim, width_size, depth, norm_scale, *, key):
        dyn_key, htb_key, hvec_key = jr.split(key, 3)

        self.hidden_dyn = Dynamics(hidden_dim, width_size, depth, key=dyn_key)
        self.hidden_vec = 0.01 * jr.normal(hvec_key, (hidden_dim,))
        self.hidden_to_eta = eqx.nn.Linear(hidden_dim, 1, key=htb_key)
        self.norm_scale = tuple(norm_scale.tolist())

        self.y0 = jnp.array([y0[0], 0, y0[1]])

    def get_eta(self, h):
        eta = jnn.sigmoid(self.hidden_to_eta(h))
        return eta.squeeze()

    def RHS(self, t, y, args=None):
        norm_state, h = y

        scale = jnp.array(self.norm_scale)
        state = norm_state * scale
        Tu, Ti, V = state

        ee = self.get_eta(h)

        ll, d, dd, N, c = 10.0, 0.01, 0.7, 100, 13.0

        dTu = ll - d*Tu - ee*V*Tu
        dTi = ee*V*Tu - dd*Ti
        dV = N*dd*Ti - c*V

        dstate = jnp.array([dTu, dTi, dV])
        dnorm_state = dstate / scale

        dh = self.hidden_dyn(t, jnp.concatenate([h, norm_state]), args)

        return (dnorm_state, dh)
    
    def loss(self, ts, ys):
        pred, _ = self.__call__(ts)

        T_pred = pred[:, 0] + pred[:, 1]
        V_pred = pred[:, 2]

        T_target = ys[:, 0] / self.norm_scale[0]
        V_target = ys[:, 1] / self.norm_scale[2]

        loss = jnp.mean(jnp.square(T_pred - T_target)) \
             + jnp.mean(jnp.square(V_pred - V_target))
        return loss
    
    def eval(self, ts_eval):
        ys_pred, h_pred = self.__call__(ts_eval)
        ys_pred = ys_pred * jnp.array(self.norm_scale)

        eta_pred = jax.vmap(self.get_eta)(h_pred)

        return ys_pred, eta_pred

    def __call__(self, ts):
        h0 = self.hidden_vec
        y0 = self.y0

        scale = jnp.array(self.norm_scale)
        norm_y0 = y0 / scale

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.RHS),
            diffrax.Tsit5(),
            # diffrax.Kvaerno5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=0.001,
            y0=(norm_y0, h0),
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            max_steps=50000,
        )

        norm_states, h = sol.ys

        return norm_states, h
