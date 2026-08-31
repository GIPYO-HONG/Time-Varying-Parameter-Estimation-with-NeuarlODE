from run import run as rn
from .base_eval import eval

if __name__=="__main__":
    fig = eval(rn)
    fig.savefig("eval/eval_figure.png", dpi=300, bbox_inches="tight")