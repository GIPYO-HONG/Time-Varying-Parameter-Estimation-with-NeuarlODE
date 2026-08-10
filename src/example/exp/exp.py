import jax.random as jr
import matplotlib.pyplot as plt
from .base_exp import BaseExperiment
from .model import Main

class Experiment(BaseExperiment):

    def __init__(self, dim, y0, ts, ys, num=5, width_size=128, depth=8, **kwargs):

        seed = kwargs.get("seed", 5678)

        model = Main(
            dim,
            num,
            width_size,
            depth,
            key=jr.PRNGKey(seed),
        )

        self.y0 = y0

        super().__init__(model, ts, ys, base_dir = "results", **kwargs)

    def loss_fn(self, model, ts, ys):
        return model.loss(self.y0, ts, ys)

    def eval(self, ts_eval):
        return self.model.eval(self.y0, ts_eval)