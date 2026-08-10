from run import run2 as rn
from .base_eval import eval
import matplotlib.pyplot as plt

ts_data, ys_data, ts_eval, ys_pred, pp_pred = eval(rn)

def plotting():
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(ts_data, ys_data, ".", label="Data")
    axs[0].plot(ts_eval, ys_pred, "--", label="Pred")
    axs[0].legend()

    axs[1].plot(ts_eval, pp_pred.squeeze(), label="param", linestyle="--")
    axs[1].legend()

    fig.savefig("eval/run2_figure.png", dpi=300, bbox_inches="tight")

if __name__=="__main__":
    plotting()