import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from utiles import BaseExperiment


class Experiment(BaseExperiment):

    def __init__(self, model_cls, y0, ts, ys, norm_scale,hidden_dim=8, width_size=32, depth=2, **kwargs):

        seed = kwargs.get("seed", 5678)

        model = model_cls(
            y0,
            hidden_dim,
            width_size,
            depth,
            norm_scale,
            key=jr.PRNGKey(seed),
        )

        super().__init__(model, ts, ys, base_dir = "HIV/results", **kwargs)

    def loss_fn(self, model, ts, ys):
        return model.loss(ts, ys)

    def eval(self, ts_eval):
        return self.model.eval(ts_eval)

    def plotting(self, ts_eval, loss_list=None):
        ys_pred, eta_pred = self.eval(ts_eval)

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        axs[0][0].plot(ts_eval, ys_pred[:, 0], "--", label="Pred T_U")
        axs[0][0].legend()

        axs[0][1].plot(ts_eval, ys_pred[:, 1], "--", label="Pred T_I")
        axs[0][1].legend()

        axs[0][2].plot(ts_eval, ys_pred[:, 0] + ys_pred[:, 1], "--", label="Pred Total T")
        axs[0][2].plot(self.ts, self.ys[:,0], ".", label="Data Total T")
        axs[0][2].legend()

        axs[1][0].plot(ts_eval, ys_pred[:, 2], "--", label="Pred V")
        axs[1][0].plot(self.ts, self.ys[:,1], ".", label="Data Total V")
        axs[1][0].set_yscale("log")
        axs[1][0].legend()

        axs[1][1].plot(ts_eval, eta_pred, label="Pred eta", linestyle="--")
        # axs[1][1].set_ylim(-1e-4, 1e-4)
        axs[1][1].legend()

        axs[1][2].plot(loss_list)
        axs[1][2].set_yscale("log")
        axs[1][2].set_title("Training Loss")
