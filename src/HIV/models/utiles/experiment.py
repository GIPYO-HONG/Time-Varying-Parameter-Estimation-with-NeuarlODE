import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import time
import os
import numpy as np
from tqdm import tqdm
from .logger import make_logger
from .optim_list import *

class BaseExperiment:
    """
    Base training experiment class.
    """

    def __init__(self, model, ts, ys, exp_name, base_dir="results", seed=5678):
        #-------- path define --------
        self.exp_dir = os.path.join(base_dir, exp_name)
        self.ckpt_dir = os.path.join(self.exp_dir, "model_parameter")
        self.log_path = os.path.join(self.exp_dir, "train.log")
        self.loss_path = os.path.join(self.exp_dir, "loss_list.npy")

        os.makedirs(self.ckpt_dir, exist_ok=True)
        #-----------------------------

        # save training log
        self.logger = make_logger(exp_name, self.log_path)

        # save data
        self.model = model
        self.ts = ts
        self.ys = ys
        self.loss_list = []

    # save model checkpoint
    def save_model(self, step):
        path = os.path.join(self.ckpt_dir, f"model_step_{step:05d}.eqx")
        eqx.tree_serialise_leaves(path, self.model)

    def loss_fn(self, model, ts, ys):
        """
        You must define loss function in each model code.
        """
        raise NotImplementedError("loss function must return (loss, aux_data)")
    
    # training loop
    def train(self, optimizer=adamw_clipping, lr=1e-3, steps=10000, viz_loss=1000):
        params, static = eqx.partition(self.model, eqx.is_inexact_array)
        
        optim = optimizer(lr)
        opt_state = optim.init(params)
        
        ts, ys = self.ts, self.ys

        # output : (loss,aux), grad
        @eqx.filter_value_and_grad
        def grad_loss(params, static, ts, ys):
            model = eqx.combine(params, static)
            return self.loss_fn(model, ts, ys)
        
        # perform 1 epoch
        def step_fn(carry, _):
            params, opt_state = carry
            loss, grads = grad_loss(params, static, ts, ys)
            updates, opt_state = optim.update(grads, opt_state, params)
            params = eqx.apply_updates(params, updates)
            # jax.debug.print("Gradients: {x}", x=grads)
            return (params, opt_state), loss
        
        # Use scan for fast training
        @eqx.filter_jit
        def train_scan(params, opt_state, n_steps):
            init_val = (params, opt_state)
            (params, opt_state), losses = jax.lax.scan(
                step_fn, init_val, None, length=n_steps
            )
            return params, opt_state, losses
        
        num_cycles = steps // viz_loss #for visualize loss decay

        #-------- Training start --------
        self.logger.info(f"start Training: {steps} steps")

        train_start = time.time()

        for cycle in tqdm(range(num_cycles), desc="Training", ncols=100):
            t0 = time.time()

            params, opt_state, batch_losses = train_scan(
                params, opt_state, viz_loss
            )

            
            self.loss_list.extend(batch_losses.tolist())

            self.model = eqx.combine(params, static)

            current_step = (cycle+1)*viz_loss
            self.save_model(current_step)

            t1 = time.time()
            msg = f"step: {current_step:5d}, loss: {batch_losses[-1]:.6e}, time: {t1 - t0:.2f}s"
            tqdm.write(msg)
            self.logger.info(msg)

        total_time = time.time() - train_start
        self.logger.info(f"Total time: {total_time/60:.2f} min")

        np.save(self.loss_path, np.array(self.loss_list))

        self.model = eqx.combine(params, static)

