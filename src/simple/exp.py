import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from utiles import BaseExperiment


class Experiment(BaseExperiment):

    def __init__(self, model_cls, y0, ts, ys, width_size=128, depth=8, **kwargs):

        seed = kwargs.get("seed", 5678)

        model = model_cls(
            width_size,
            depth,
            key=jr.PRNGKey(seed),
        )

        self.y0 = y0

        super().__init__(model, ts, ys, base_dir = "simple/results", **kwargs)

    def loss_fn(self, model, ts, ys):
        return model.loss(self.y0, ts, ys)

    def eval(self, ts_eval):
        return self.model.eval(self.y0, ts_eval)

    def plotting(self, ts_eval, loss_list=None):
        ys_pred, pp_pred = self.eval(ts_eval)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].plot(self.ts, self.ys, ".", label="Data")
        axs[0].plot(ts_eval, ys_pred, "--", label="Pred")
        axs[0].legend()

        axs[1].plot(ts_eval, pp_pred, label="param", linestyle="--")
        axs[1].legend()
