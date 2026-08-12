from data.data import make_data
from exp.exp_tmp import Experiment
from exp.models import *

exp_name = "run1"

model = no

patid = 16 #1~46
ts, ys, y0, scale = make_data(patid)

param = [10, 1000, 13]

EX = Experiment(
    model_cls = model.Main,
    y0=y0,
    param=param,
    ts=ts,
    ys=ys,
    norm_scale=scale,
    exp_name=exp_name,
)

if __name__=="__main__":
    EX.train(lr=1e-5, steps=10000, lbfgs_steps=0)