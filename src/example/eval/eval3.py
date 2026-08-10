from run import run3 as rn
from .base_eval import eval
import matplotlib.pyplot as plt

ts_data, ys_data, ts_eval, ys_pred, pp_pred = eval(rn)

def plotting():
    aa, bb, dd, gg = pp_pred.reshape(pp_pred.shape[0], -1).T

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(ts_data, ys_data, ".", label="Data")
    axs[0].plot(ts_eval, ys_pred, "--", label="Pred")
    axs[0].legend()

    axs[1].plot(ts_eval, aa, label="param alpha", linestyle="--")
    axs[1].plot(ts_eval, bb, label="param beta", linestyle="--")
    axs[1].plot(ts_eval, dd, label="param delta", linestyle="--")
    axs[1].plot(ts_eval, gg, label="param gamma", linestyle="--")
    axs[1].legend()

    fig.savefig("eval/eval3_figure.png", dpi=300, bbox_inches="tight")

if __name__=="__main__":
    plotting()