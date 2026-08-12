import jax
import jax.numpy as jnp
import equinox as eqx
import time
import os
import numpy as np
from tqdm import tqdm
import optax
import jaxopt
from .logger import make_logger

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
        self.best_parameter_path = os.path.join(self.ckpt_dir, "best_parameter.eqx")
        self.best_parameter = None
        self.best_loss = float("inf")

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
    def train(self, lr=1e-3, steps=10000, viz_loss=1000, lbfgs_steps=500, lbfgs_viz_loss=50, lbfgs_memory_size=10):
        """Train with AdamW, then optionally refine with JAXopt L-BFGS.

        ``best_parameter`` is the parameter tree with the smallest observed
        training loss across both optimisation stages.  It is also written to
        ``model_parameter/best_parameter.eqx``.
        """
        if steps < 0 or lbfgs_steps < 0:
            raise ValueError("steps and lbfgs_steps must be non-negative")
        if viz_loss < 1:
            raise ValueError("viz_loss must be positive")
        if lbfgs_viz_loss is None:
            lbfgs_viz_loss = viz_loss
        if lbfgs_viz_loss < 1:
            raise ValueError("lbfgs_viz_loss must be positive")

        params, static = eqx.partition(self.model, eqx.is_inexact_array)
        ts, ys = self.ts, self.ys
        best_params = params
        best_loss = jnp.array(jnp.inf)

        def loss_value(trainable):
            model = eqx.combine(trainable, static)
            return self.loss_fn(model, ts, ys)

        @eqx.filter_value_and_grad
        def grad_loss(trainable):
            return loss_value(trainable)

        def keep_best(candidate, current, is_better):
            return jax.tree.map(
                lambda new, old: jnp.where(is_better, new, old), candidate, current
            )

        #-------- AdamW training --------
        adamw = optax.adamw(lr)
        adamw_state = adamw.init(params)

        def adamw_step(carry, _):
            trainable, opt_state, current_best, current_best_loss = carry
            loss, grads = grad_loss(trainable)
            is_better = loss < current_best_loss
            current_best = keep_best(trainable, current_best, is_better)
            current_best_loss = jnp.where(is_better, loss, current_best_loss)
            updates, opt_state = adamw.update(grads, opt_state, trainable)
            trainable = eqx.apply_updates(trainable, updates)
            return (trainable, opt_state, current_best, current_best_loss), loss

        @eqx.filter_jit
        def adamw_scan(trainable, opt_state, current_best, current_best_loss, n_steps):
            return jax.lax.scan(
                adamw_step,
                (trainable, opt_state, current_best, current_best_loss),
                None,
                length=n_steps,
            )

        self.logger.info(f"start AdamW training: {steps} steps")
        train_start = time.time()
        completed_steps = 0
        for start in tqdm(range(0, steps, viz_loss), desc="AdamW", ncols=100):
            block_steps = min(viz_loss, steps - start)
            t0 = time.time()
            (params, adamw_state, best_params, best_loss), batch_losses = adamw_scan(
                params, adamw_state, best_params, best_loss, block_steps
            )
            self.loss_list.extend(np.asarray(batch_losses).tolist())
            self.model = eqx.combine(params, static)
            completed_steps += block_steps
            self.save_model(completed_steps)
            msg = f"AdamW step: {completed_steps:5d}, loss: {batch_losses[-1]:.6e}, time: {time.time() - t0:.2f}s"
            tqdm.write(msg)
            self.logger.info(msg)

        # The final AdamW update is not evaluated inside its update step.
        final_adamw_loss = loss_value(params)
        is_better = final_adamw_loss < best_loss
        best_params = keep_best(params, best_params, is_better)
        best_loss = jnp.where(is_better, final_adamw_loss, best_loss)

        # Start L-BFGS from AdamW's best model, not its last update.
        params = best_params
        self.model = eqx.combine(params, static)
        self.logger.info(f"AdamW best loss: {float(best_loss):.6e}")

        #-------- L-BFGS refinement --------
        if lbfgs_steps:
            lbfgs = jaxopt.LBFGS(
                fun=loss_value,
                maxiter=lbfgs_steps,
                history_size=lbfgs_memory_size,
                implicit_diff=False,
            )
            lbfgs_state = lbfgs.init_state(params)

            def lbfgs_step(carry, _):
                trainable, solver_state, current_best, current_best_loss = carry
                loss = loss_value(trainable)
                is_better = loss < current_best_loss
                current_best = keep_best(trainable, current_best, is_better)
                current_best_loss = jnp.where(is_better, loss, current_best_loss)
                result = lbfgs.update(trainable, solver_state)
                return (
                    result.params,
                    result.state,
                    current_best,
                    current_best_loss,
                ), loss

            @eqx.filter_jit
            def lbfgs_scan(trainable, solver_state, current_best,
                           current_best_loss, n_steps):
                return jax.lax.scan(
                    lbfgs_step,
                    (trainable, solver_state, current_best, current_best_loss),
                    None,
                    length=n_steps,
                )

            self.logger.info(f"start L-BFGS refinement: {lbfgs_steps} steps")
            completed_lbfgs_steps = 0
            for start in tqdm(
                range(0, lbfgs_steps, lbfgs_viz_loss), desc="L-BFGS", ncols=100
            ):
                block_steps = min(lbfgs_viz_loss, lbfgs_steps - start)
                t0 = time.time()
                (params, lbfgs_state, best_params, best_loss), lbfgs_losses = lbfgs_scan(
                    params, lbfgs_state, best_params, best_loss, block_steps
                )
                self.loss_list.extend(np.asarray(lbfgs_losses).tolist())
                self.model = eqx.combine(params, static)
                completed_lbfgs_steps += block_steps
                self.save_model(steps + completed_lbfgs_steps)
                msg = (
                    f"L-BFGS step: {completed_lbfgs_steps:5d}, "
                    f"loss: {lbfgs_losses[-1]:.6e}, time: {time.time() - t0:.2f}s"
                )
                tqdm.write(msg)
                self.logger.info(msg)


            # Include the parameter produced by the final L-BFGS update.
            final_lbfgs_loss = loss_value(params)
            is_better = final_lbfgs_loss < best_loss
            best_params = keep_best(params, best_params, is_better)
            best_loss = jnp.where(is_better, final_lbfgs_loss, best_loss)

        self.best_parameter = eqx.combine(best_params, static)
        self.best_loss = float(best_loss)
        eqx.tree_serialise_leaves(self.best_parameter_path, self.best_parameter)
        self.model = self.best_parameter
        np.save(self.loss_path, np.asarray(self.loss_list))

        total_time = time.time() - train_start
        self.logger.info(f"best loss: {self.best_loss:.6e}")
        self.logger.info(f"Total time: {total_time/60:.2f} min")
