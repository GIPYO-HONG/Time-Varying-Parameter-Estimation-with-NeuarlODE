from data.data import make_data
from exp.exp import Experiment
import exp.model as model

exp_name = "run"

ts, ys, days = make_data("20200124", "20231231")
# ts, ys, days = make_data("20200124", "20211231")

EX = Experiment(
    model_cls = model.Main,
    ts=ts,
    ys=ys,
    days=days,
    exp_name=exp_name,
)

if __name__=="__main__":
    EX.train(lr=1e-5, steps=500000, lbfgs_steps=0)