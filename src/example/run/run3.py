from data.data_generate import data_generate as dg
from data.func_list import *
from exp.exp import Experiment

exp_name = "run3"

y0, ts, ys, dim = dg(exp3)

EX = Experiment(
    dim=dim,
    y0=y0,
    ts=ts,
    ys=ys,
    exp_name=exp_name,
)

if __name__=="__main__":
    EX.train(lr=1e-4, steps=100000, lbfgs_steps=0)
