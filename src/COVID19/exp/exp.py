import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from .base_exp import BaseExperiment


class Experiment(BaseExperiment):

    def __init__(self, model_cls, ts, ys, days, width_size=64, depth=8, **kwargs):

        seed = kwargs.get("seed", 5678)

        model = model_cls(
            width_size,
            depth,
            days,
            key=jr.PRNGKey(seed),
        )

        super().__init__(model, ts, ys, base_dir = "results", **kwargs)

    def loss_fn(self, model, ts, ys):
        return model.loss(ts, ys)

    def relative_error(self, pred, true):
        return jnp.linalg.norm(pred - true, 2) / jnp.linalg.norm(true, 2)

    def eval(self, ts_eval):
        return self.model.eval(ts_eval)

    def plotting(self, ts_eval, loss_list=None):
        ys_pred, beta_pred = self.eval(ts_eval)

        fig, axs = plt.subplots(2, 1, figsize=(15, 5))

        axs[0].plot(self.ts, self.ys.squeeze(), ".", label="DATA")
        axs[0].plot(ts_eval, 0.25*ys_pred[:, 1], "--", label="PREDICTION")
        axs[0].legend()

        axs[1].plot(ts_eval, beta_pred, label="Pred beta", linestyle="--")
        axs[1].legend()

        return fig
