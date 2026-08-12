from run import run2 as rn
from .base_eval import eval

if __name__=="__main__":
    fig = eval(rn)
    fig.savefig("eval/eval2_figure.png", dpi=300, bbox_inches="tight")