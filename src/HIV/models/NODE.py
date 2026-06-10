########## module import ##########
import jax
import jax.random as jr
import jax.numpy as jnp
import jax.nn as jnn

import equinox as eqx
import diffrax

#BaseExperiment, make_logger, beta_generate, get_data, SEIAR, plotting
from .utiles import *


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
    
class NODE(eqx.Module):
    eta: Eta
    norm_scale: tuple = eqx.field(static=True)

    parameter: jnp.ndarray
    y0: jnp.ndarray

    def __init__(self, width_size, depth, norm_scale, *, key):

        self.norm_scale = tuple(norm_scale.tolist())

        self.eta = Eta(width_size, depth, key=key)

        def softplus_inv(x):
            # softplus(y) ≈ y for large y → inv ≈ identity
            # 수치 안전한 버전: log(exp(x) - 1) = x + log(1 - exp(-x))
            return jnp.where(
                x > 20.0,
                x,  # large x: softplus_inv(x) ≈ x
                jnp.log(jnp.expm1(jnp.clip(x, 1e-6, 20.0)))
            )

        targets_param = jnp.array([44.21, 0.11767, 1093.4, 0.5535, 3.0657])
        self.parameter = softplus_inv(targets_param)

        targets_y0 = jnp.array([580, 20, 10**5])
        self.y0 = softplus_inv(targets_y0)

    def RHS(self, t, y, args=None):

        Tu, Ti, V = y

        ee = self.eta(t)

        ll, rr, N, dd, c = jnn.softplus(self.parameter)

        dTu = ll - rr * Tu - ee * Tu * V
        dTi = ee * Tu * V - dd * Ti
        dV  = N * dd * Ti - c * V

        dstate = jnp.array([dTu, dTi, dV])

        return dstate
    
    def RHS(self, t, y, args=None):
        norm_state = y

        scale = jnp.array(self.norm_scale)
        state = norm_state * scale
        Tu, Ti, V = state

        ee = self.eta(t)

        ll, rr, N, dd, c = jnn.softplus(self.parameter)

        dTu = ll - rr * Tu - ee * Tu * V
        dTi = ee * Tu * V - dd * Ti
        dV  = N * dd * Ti - c * V

        dstate = jnp.array([dTu, dTi, dV])
        dnorm_state = dstate / scale

        return dnorm_state
    
    
    
    def __call__(self, ts):
        y0 = jnn.softplus(self.y0)
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
    
########## Experiment ##########
    
class Experiment(BaseExperiment):

    def __init__(self, ts, ys, eta, width_size=64, depth=4, **kwargs):

        seed = kwargs.get("seed", 5678)

        scales_obs = jnp.max(ys, axis=0) + 1e-6  # shape: (2,), [T_max, V_max]
        self.scales = scales_obs

        self.norm_scale = jnp.array([scales_obs[0], scales_obs[0], scales_obs[1]])

        model = NODE(
            width_size,
            depth,
            norm_scale=self.norm_scale,
            key=jax.random.PRNGKey(seed),
        )

        super().__init__(model, ts, ys, **kwargs)

        self.eta = eta

    def loss_fn(self, model, ts, ys):
        pred = model(ts)

        T_pred = pred[:, 0] + pred[:, 1]
        V_pred = pred[:, 2]

        T_target = ys[:, 0] / self.scales[0]
        V_target = ys[:, 1] / self.scales[1]

        loss = jnp.mean(jnp.square(T_pred - T_target)) \
             + jnp.mean(jnp.square(V_pred - V_target))
        return loss
    
########## Evaluation ##########

def Evaluation(EX, y0_, ts_eval, loss_list, viz_data=False):
    ts_data, ys_data, eta, model, scales = EX.ts, EX.ys, EX.eta, EX.model, EX.norm_scale

    ys_eval  = get_data(ts_eval, y0_, eta)
    ys_pred = model(ts_eval)
    ys_pred = ys_pred * scales

    eta_eval = EX.eta(ts_eval)
    eta_pred = jax.vmap(lambda t: model.eta(jnp.array([t])))(ts_eval)

    plotting(ts_data, ys_data, ts_eval, ys_eval, ys_pred, eta_eval, eta_pred, loss_list, viz_data)