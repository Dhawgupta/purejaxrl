import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
from flax.training.train_state import TrainState
import distrax
import gymnax
from jax.flatten_util import ravel_pytree
from wrappers import LogWrapper, FlattenObservationWrapper


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        # Actor network.
        actor_mean = nn.Dense(
            8, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            8, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # # Critic network (not used in IS objective, but computed).
        # critic = nn.Dense(
        #     64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        # )(x)
        # critic = activation(critic)
        # critic = nn.Dense(
        #     64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        # )(critic)
        # critic = activation(critic)
        # critic = nn.Dense(
        #     1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        # )(critic)

        return pi, 0
        # return pi, jnp.squeeze(critic, axis=-1)



def compute_log_rho(params_flat, batch, network, unravel_fn):
    NUM_ENVS, T, obs_dim = batch["obs"].shape
    obs_flat = batch["obs"].reshape(-1, obs_dim)
    # Recompute current log probabilities for all observations.
    params = unravel_fn(params_flat)
    pi_flat, _ = network.apply(params, obs_flat)
    current_log_probs_flat = pi_flat.log_prob(batch["actions"].reshape(-1))
    current_log_probs = current_log_probs_flat.reshape(NUM_ENVS, T)
    # Compute per-trajectory IS ratio:
    # ratio = exp(sum_{t=0}^{T-1} (current_log_prob - stored_log_prob) * mask)
    # OUTPUT: [ NUM_ENVS, ]
    log_ratio = jnp.sum(
        (current_log_probs - jax.lax.stop_gradient(batch["stored_log_probs"])  ) * batch["masks"], axis=1
                )
    return log_ratio

def compute_objective(params_flat, batch, network, unravel_fn):
    log_rho = compute_log_rho(params_flat, batch, network, unravel_fn)
    rho = jnp.exp(log_rho)
    G0 = batch["returns"][:, 0] # [NUM_ENVS, ]
    obj = + jnp.mean(rho * G0) # the objective values
    return obj

def compute_constraint_kl(params_flat, batch, network, unravel_fn):
    log_rho = compute_log_rho(params_flat, batch, network, unravel_fn)
    return -jnp.mean(log_rho) # ()

def compute_sample_mean(params_flat, batch, network, unravel_fn):
    log_rho = compute_log_rho(params_flat, batch, network, unravel_fn)
    rho = jnp.exp(log_rho)
    return jnp.mean(rho) # ()

grad_objective = jax.grad(compute_objective, argnums=0)
grad_kl = jax.grad(compute_constraint_kl, argnums=0)
grad_mean = jax.grad(compute_sample_mean, argnums=0)

def dual_ascent_linear_obj_linear_con_all_jit(params_flat, dataset, network, unravel_fn, 
                                                delta=0.4, epsilon=0.05,
                                                tau=1.0, alpha_dual=1e-2, tol=1e-4,
                                                max_dual_iters=50):
    """
    Computes an update s such that the new policy = policy + s satisfies
    the following linearly approximated constraints:
      1) c1 + d1^T s <= delta,
      2) 1-epsilon <= c2 + d2^T s <= 1+epsilon,
    while maximizing the linearized objective with quadratic regularization:
      max_s   g^T s - (1/(2*tau)) ||s||^2.
    The unconstrained optimum is s0 = tau * g.
    If s0 violates any constraint, dual ascent is performed.
    """
    # First-order quantities at the current parameters.
    g      = grad_objective(params_flat, dataset, network, unravel_fn)
    c1_val = compute_constraint_kl(params_flat, dataset, network, unravel_fn)
    c2_val = compute_sample_mean(params_flat, dataset, network, unravel_fn)
    d1     = grad_kl(params_flat, dataset, network, unravel_fn)
    d2     = grad_mean(params_flat, dataset, network, unravel_fn)

    # Unconstrained update.
    s0 = tau * g

    # Linear approximations of the constraints at s0.
    c1_approx0 = c1_val + jnp.dot(d1, s0)
    c2_approx0 = c2_val + jnp.dot(d2, s0)

    unconstrained_feasible = (c1_approx0 <= delta) & (((1 - epsilon) <= c2_approx0) & (c2_approx0 <= (1 + epsilon)))
    
    def dual_cond(state):
        i, lambda1, lambda2, lambda3, s = state
        c1_approx = c1_val + jnp.dot(d1, s)
        c2_approx = c2_val + jnp.dot(d2, s)
        v1 = jnp.maximum(0.0, c1_approx - delta)
        v2 = jnp.maximum(0.0, c2_approx - (1 + epsilon))
        v3 = jnp.maximum(0.0, (1 - epsilon) - c2_approx)
        violation = (v1 >= tol) | (v2 >= tol) | (v3 >= tol)
        return violation & (i < max_dual_iters)
    
    def dual_body(state):
        i, lambda1, lambda2, lambda3, s = state
        # With linear approximations, the optimal update is given in closed form.
        s_new = tau * (g - lambda1 * d1 - (lambda2 - lambda3) * d2)
        c1_approx = c1_val + jnp.dot(d1, s_new)
        c2_approx = c2_val + jnp.dot(d2, s_new)
        lambda1_new = jnp.maximum(0.0, lambda1 + alpha_dual * (c1_approx - delta))
        lambda2_new = jnp.maximum(0.0, lambda2 + alpha_dual * (c2_approx - (1 + epsilon)))
        lambda3_new = jnp.maximum(0.0, lambda3 + alpha_dual * ((1 - epsilon) - c2_approx))
        return (i + 1, lambda1_new, lambda2_new, lambda3_new, s_new)
    
    def dual_ascent_loop():
        init_state = (0, 0.0, 0.0, 0.0, s0)
        final_state = jax.lax.while_loop(dual_cond, dual_body, init_state)
        return final_state[4]
    
    s_final = jax.lax.cond(unconstrained_feasible,
                           lambda _: s0,
                           lambda _: dual_ascent_loop(),
                           operand=None)
    return s_final

def dual_ascent_linear_obj_linear_con_all_quasi_jit(params_flat, dataset, network, unravel_fn, 
                                                     delta=0.4, epsilon=0.05,
                                                     tau=1.0, tol=1e-4,
                                                     max_dual_iters=50):
    """
    Same as before but using a quasi-Newton update (e.g. BFGS) for the dual variables.
    Here we treat the dual variables as a vector λ = [λ₁, λ₂, λ₃] and update them via
      λ_new = max(0, λ + H_inv * grad_dual),
    where grad_dual contains the linearized constraint violations.
    """
    # First-order quantities at current parameters.
    g      = grad_objective(params_flat, dataset, network, unravel_fn)
    c1_val = compute_constraint_kl(params_flat, dataset, network, unravel_fn)
    c2_val = compute_sample_mean(params_flat, dataset, network, unravel_fn)
    d1     = grad_kl(params_flat, dataset, network, unravel_fn)
    d2     = grad_mean(params_flat, dataset, network, unravel_fn)

    s0 = tau * g
    # Linear approximations of the constraints:
    c1_approx0 = c1_val + jnp.dot(d1, s0)
    c2_approx0 = c2_val + jnp.dot(d2, s0)

    unconstrained_feasible = (c1_approx0 <= delta) & (((1 - epsilon) <= c2_approx0) & (c2_approx0 <= (1 + epsilon)))
    
    # We'll update dual variables λ = [λ₁, λ₂, λ₃] using a quasi-Newton update.
    # Initialize λ and an identity matrix for H_inv.
    init_lambda = jnp.zeros(3)
    H_inv = jnp.eye(3)  # initial inverse Hessian approximation
    
    def dual_cond(state):
        i, lambd, s, H_inv = state
        c1_approx = c1_val + jnp.dot(d1, s)
        c2_approx = c2_val + jnp.dot(d2, s)
        v1 = jnp.maximum(0.0, c1_approx - delta)
        v2 = jnp.maximum(0.0, c2_approx - (1 + epsilon))
        v3 = jnp.maximum(0.0, (1 - epsilon) - c2_approx)
        violation = (v1 >= tol) | (v2 >= tol) | (v3 >= tol)
        return violation & (i < max_dual_iters)
    
    def dual_body(state):
        i, lambd, s, H_inv = state
        # Compute the dual gradient vector:
        # grad_dual = [c1_approx - delta, c2_approx - (1+epsilon), (1-epsilon) - c2_approx]
        c1_approx = c1_val + jnp.dot(d1, s)
        c2_approx = c2_val + jnp.dot(d2, s)
        grad_dual = jnp.array([c1_approx - delta, c2_approx - (1 + epsilon), (1 - epsilon) - c2_approx])
        
        # Quasi-Newton update for the dual variables.
        step = H_inv @ grad_dual
        lambd_new = jnp.maximum(0.0, lambd + step)
        
        # Compute new s with updated duals.
        # Here, s = τ * (g - λ₁ d₁ - (λ₂ - λ₃) d₂)
        s_new = tau * (g - lambd_new[0] * d1 - (lambd_new[1] - lambd_new[2]) * d2)
        
        # For the BFGS update, we need to compute:
        # y = grad_dual_new - grad_dual, and p = lambd_new - lambd.
        # For simplicity, assume we can compute grad_dual_new similarly.
        c1_approx_new = c1_val + jnp.dot(d1, s_new)
        c2_approx_new = c2_val + jnp.dot(d2, s_new)
        grad_dual_new = jnp.array([c1_approx_new - delta, c2_approx_new - (1 + epsilon), (1 - epsilon) - c2_approx_new])
        p = lambd_new - lambd
        y = grad_dual_new - grad_dual
        
        # Update H_inv using the BFGS formula:
        # H_inv_new = (I - p y^T / (y^T p)) H_inv (I - y p^T / (y^T p)) + (p p^T) / (y^T p)
        denom = jnp.dot(y, p) + 1e-8
        I_dual = jnp.eye(3)
        H_inv_new = (I_dual - jnp.outer(p, y) / denom) @ H_inv @ (I_dual - jnp.outer(y, p) / denom) + jnp.outer(p, p) / denom
        
        return (i + 1, lambd_new, s_new, H_inv_new)
    
    def dual_ascent_loop():
        init_state = (0, init_lambda, s0, H_inv)
        final_state = jax.lax.while_loop(dual_cond, dual_body, init_state)
        return final_state[2]  # return s

    s_final = jax.lax.cond(unconstrained_feasible,
                           lambda _: s0,
                           lambda _: dual_ascent_loop(),
                           operand=None)
    return s_final


def dual_ascent_linear_obj_quadratic_con_all_jit(params_flat, dataset, network, unravel_fn, 
                                                 delta=0.4, epsilon=0.05,
                                                 tau=1.0, alpha_dual=1e-2, tol=1e-4,
                                                 max_dual_iters=50, damping=1e-3):
    """
    Computes an update s such that the new policy = policy + s satisfies
    the following approximated constraints:
      1) c1 + d1^T s + 0.5 s^T H_kl s <= delta,
      2) 1-epsilon <= c2 + d2^T s + 0.5 s^T H_mean s <= 1+epsilon,
    while maximizing the linearized objective with quadratic regularization:
      max_s   g^T s - (1/(2*tau)) ||s||^2.
    The unconstrained optimum is s0 = tau * g.
    If s0 violates any constraint, dual ascent is performed.
    """
    g      = grad_objective(params_flat, dataset, network, unravel_fn)
    c1_val = compute_constraint_kl(params_flat, dataset, network, unravel_fn)
    c2_val = compute_sample_mean(params_flat, dataset, network, unravel_fn)
    d1     = grad_kl(params_flat, dataset, network, unravel_fn)
    d2     = grad_mean(params_flat, dataset, network, unravel_fn)

    H_kl   = jax.hessian(compute_constraint_kl, argnums=0)(params_flat, dataset, network, unravel_fn)
    H_mean = jax.hessian(compute_sample_mean, argnums=0)(params_flat, dataset, network, unravel_fn)

    s0 = tau * g

    c1_approx0 = c1_val + jnp.dot(d1, s0) + 0.5 * jnp.dot(s0, jnp.dot(H_kl, s0))
    c2_approx0 = c2_val + jnp.dot(d2, s0) + 0.5 * jnp.dot(s0, jnp.dot(H_mean, s0))

    unconstrained_feasible = (c1_approx0 <= delta) & (((1 - epsilon) <= c2_approx0) & (c2_approx0 <= (1 + epsilon)))
    I = jnp.eye(params_flat.shape[0])

    def dual_cond(state):
        i, lambda1, lambda2, lambda3, s = state
        c1_approx = c1_val + jnp.dot(d1, s) + 0.5 * jnp.dot(s, jnp.dot(H_kl, s))
        c2_approx = c2_val + jnp.dot(d2, s) + 0.5 * jnp.dot(s, jnp.dot(H_mean, s))
        v1 = jnp.maximum(0.0, c1_approx - delta)
        v2 = jnp.maximum(0.0, c2_approx - (1 + epsilon))
        v3 = jnp.maximum(0.0, (1 - epsilon) - c2_approx)
        violation = (v1 >= tol) | (v2 >= tol) | (v3 >= tol)
        return violation & (i < max_dual_iters)

    def dual_body(state):
        i, lambda1, lambda2, lambda3, s = state
        A = (1.0 / tau) * I + lambda1 * H_kl + (lambda2 - lambda3) * H_mean + damping * I
        b = g - lambda1 * d1 - (lambda2 - lambda3) * d2
        s_new = jnp.linalg.solve(A, b)
        c1_approx = c1_val + jnp.dot(d1, s_new) + 0.5 * jnp.dot(s_new, jnp.dot(H_kl, s_new))
        c2_approx = c2_val + jnp.dot(d2, s_new) + 0.5 * jnp.dot(s_new, jnp.dot(H_mean, s_new))
        lambda1_new = jnp.maximum(0.0, lambda1 + alpha_dual * (c1_approx - delta))
        lambda2_new = jnp.maximum(0.0, lambda2 + alpha_dual * (c2_approx - (1 + epsilon)))
        lambda3_new = jnp.maximum(0.0, lambda3 + alpha_dual * ((1 - epsilon) - c2_approx))
        return (i + 1, lambda1_new, lambda2_new, lambda3_new, s_new)

    def dual_ascent_loop():
        init_state = (0, 0.0, 0.0, 0.0, s0)
        final_state = jax.lax.while_loop(dual_cond, dual_body, init_state)
        return final_state[4]

    s_final = jax.lax.cond(unconstrained_feasible,
                       lambda _: s0,
                       lambda _: dual_ascent_loop(),
                       operand=None)
    return s_final

def make_train(config):
    # For REINFORCE we treat each full episode as one trajectory.
    # TOTAL_EPISODES defines the number of update iterations.
    config["NUM_UPDATES"] = config["TOTAL_EPISODES"]
    env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def train(rng):
        # ---------------------------
        # Initialize Network & Optimizer.
        # ---------------------------
        network = ActorCritic(env.action_space(env_params).n, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        tx = optax.chain(
            # optax.rmsprop(config["LR"], eps=1e-5),
            optax.rmsprop(config["LR"]),
        )
        train_state = TrainState.create(
            apply_fn=network.apply, params=network_params, tx=tx
        )

        # ---------------------------
        # Initialize Environments.
        # ---------------------------
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        max_ep_len = config.get("MAX_EPISODE_LENGTH", 500)

        # ------------------------------------------------------------
        # Run a Single Episode (Trajectory) Until Termination or Timeout.
        # ------------------------------------------------------------
        def run_episode(train_state, env_state, obs, rng):
            # Preallocate fixed buffers for a full episode.
            obs_buffer = jnp.zeros((max_ep_len,) + obs.shape)
            actions_buffer = jnp.zeros((max_ep_len,), dtype=jnp.int32)
            rewards_buffer = jnp.zeros((max_ep_len,))
            log_probs_buffer = jnp.zeros((max_ep_len,))
            values_buffer = jnp.zeros((max_ep_len,))

            def cond_fun(state):
                t, env_state, obs, rng, done, *_ = state
                return jnp.logical_and(t < max_ep_len, jnp.logical_not(done))

            def body_fun(state):
                t, env_state, obs, rng, done, obs_buffer, actions_buffer, rewards_buffer, log_probs_buffer, values_buffer = state
                rng, rng_action = jax.random.split(rng)
                pi, value = network.apply(train_state.params, obs)
                action = pi.sample(seed=rng_action)
                log_prob = pi.log_prob(action)

                obs_buffer = obs_buffer.at[t].set(obs)
                actions_buffer = actions_buffer.at[t].set(action)
                log_probs_buffer = log_probs_buffer.at[t].set(log_prob)
                values_buffer = values_buffer.at[t].set(value)

                rng, rng_step = jax.random.split(rng)
                obs_next, env_state, reward, done, _ = env.step(rng_step, env_state, action, env_params)
                rewards_buffer = rewards_buffer.at[t].set(reward)

                return (
                    t + 1,
                    env_state,
                    obs_next,
                    rng,
                    done,
                    obs_buffer,
                    actions_buffer,
                    rewards_buffer,
                    log_probs_buffer,
                    values_buffer,
                )

            init_state = (
                0,
                env_state,
                obs,
                rng,
                False,
                obs_buffer,
                actions_buffer,
                rewards_buffer,
                log_probs_buffer,
                values_buffer,
            )
            final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
            t, env_state, obs, rng, done, obs_buffer, actions_buffer, rewards_buffer, log_probs_buffer, values_buffer = final_state
            return (
                {
                    "obs": obs_buffer,
                    "actions": actions_buffer,
                    "rewards": rewards_buffer,
                    "log_probs": log_probs_buffer,
                    "values": values_buffer,
                    "length": t,
                },
                env_state,
                obs,  # last observation (not used further)
                rng,
            )

        # ------------------------------------------------------------
        # Compute Discounted Returns for a Single Trajectory.
        # ------------------------------------------------------------
        def compute_returns(rewards, mask):
            def body_fun(carry, r_m):
                r, m = r_m
                new_carry = r + config["GAMMA"] * carry * m
                return new_carry, new_carry
            _, returns = jax.lax.scan(body_fun, 0.0, (rewards[::-1], mask[::-1]))
            returns = returns[::-1]
            return returns

        # Vectorize return computation over a batch (one trajectory per environment).
        v_compute_returns = jax.vmap(compute_returns, in_axes=(0, 0))

        # ------------------------------------------------------------
        # UPDATE STEP: Collect One Trajectory per Env, Compute Loss & Update.
        # ------------------------------------------------------------
        def _update_step(runner_state, unused):
            train_state, env_state, obsv, rng = runner_state
            rng, rng_epi = jax.random.split(rng)
            epi_rngs = jax.random.split(rng_epi, config["NUM_ENVS"])
            results = jax.vmap(run_episode, in_axes=(None, 0, 0, 0))(
                train_state, env_state, obsv, epi_rngs
            )
            traj_batch, new_env_state, new_obsv, rngs = results
            lengths = traj_batch["length"]  # shape: (NUM_ENVS,)

            timesteps = jnp.arange(max_ep_len)[None, :]  # shape (1, max_ep_len)
            masks = timesteps < lengths[:, None]           # shape (NUM_ENVS, max_ep_len)

            returns = v_compute_returns(traj_batch["rewards"], masks)

            # Build a batch dictionary in trajectory form.
            batch = {
                "obs": traj_batch["obs"],                # (NUM_ENVS, max_ep_len, obs_dim)
                "actions": traj_batch["actions"],          # (NUM_ENVS, max_ep_len)
                "stored_log_probs": traj_batch["log_probs"],  # (NUM_ENVS, max_ep_len)
                "masks": masks,                            # (NUM_ENVS, max_ep_len)
                "returns": returns,                        # (NUM_ENVS, max_ep_len)
            }

            def loss_fn(params, batch):
                NUM_ENVS, T, obs_dim = batch["obs"].shape
                obs_flat = batch["obs"].reshape(-1, obs_dim)
                # Recompute current log probabilities for all observations.
                pi_flat, _ = network.apply(params, obs_flat)
                current_log_probs_flat = pi_flat.log_prob(batch["actions"].reshape(-1))
                current_log_probs = current_log_probs_flat.reshape(NUM_ENVS, T)
                # Compute per-trajectory IS ratio:
                # ratio = exp(sum_{t=0}^{T-1} (current_log_prob - stored_log_prob) * mask)
                log_ratio = jnp.sum(
                    (current_log_probs - jax.lax.stop_gradient(batch["stored_log_probs"])  ) * batch["masks"], axis=1
                )
                ratio = jnp.exp(log_ratio)
                # Return from time 0 for each trajectory.
                G0 = batch["returns"][:, 0]
                # Our objective is to maximize mean (ratio * G0), so we minimize the negative.
                loss = -jnp.mean(ratio * G0)
                return loss, (loss, jnp.mean(ratio), jnp.mean(G0))
            
            total_loss = 0
            total_loss_val = 0
            total_ratio = 0
            total_G0 = 0
            
            # Perform optimization
            for _ in range(config["NUM_UPDATES_PER_BATCH"]):
                flat_params, unravel_fn = ravel_pytree(train_state.params)
                # negative because we are maximizing the objective
                # grads_flat =  - dual_ascent_linear_obj_quadratic_con_all_jit(flat_params, batch, network, unravel_fn, 
                #                                                 delta=config.get("delta", 0.4),
                #                                                 epsilon=config.get("epsilon", 0.05),
                #                                                 tau=config.get("tau", 1.0),
                #                                                 alpha_dual=config.get("alpha_dual", 1.0),
                #                                                 tol=config.get("tol", 1e-4),
                #                                                 max_dual_iters=config.get("max_dual_iters", 200),
                #                                                 damping=config.get("damping", 1e-3))
                grads_flat = - dual_ascent_linear_obj_linear_con_all_jit(flat_params, batch, network, unravel_fn, 
                                                                delta=config.get("delta", 0.4),
                                                                epsilon=config.get("epsilon", 0.05),
                                                                tau=config.get("tau", 1.0),
                                                                alpha_dual=config.get("alpha_dual", 1.0),
                                                                tol=config.get("tol", 1e-4),
                                                                max_dual_iters=config.get("max_dual_iters", 200))
                # grads_flat = - dual_ascent_linear_obj_linear_con_all_quasi_jit(flat_params, batch, network, unravel_fn,
                #                                                 delta=config.get("delta", 0.4),
                #                                                 epsilon=config.get("epsilon", 0.05),
                #                                                 tau=config.get("tau", 1.0),
                #                                                 tol=config.get("tol", 1e-4),
                #                                                 max_dual_iters=config.get("max_dual_iters", 200))
                
                grads = unravel_fn(grads_flat)
                (loss_val_, (loss_val, mean_ratio, mean_G0)) = \
                    loss_fn(train_state.params, batch)
                total_loss += loss_val_
                total_loss_val += loss_val
                total_ratio += mean_ratio
                total_G0 += mean_G0
                # negative because we are maximizing the objective
                train_state = train_state.apply_gradients(grads=grads)
            


            # for _ in range(config["NUM_UPDATES_PER_BATCH"]):
            #     (loss_val_, (loss_val, mean_ratio, mean_G0)), grads = \
            #         jax.value_and_grad(loss_fn, has_aux=True)(train_state.params, batch)
            #     total_loss += loss_val_
            #     total_loss_val += loss_val
            #     total_ratio += mean_ratio
            #     total_G0 += mean_G0
            #     train_state = train_state.apply_gradients(grads=grads)


            total_loss /= config["NUM_UPDATES_PER_BATCH"]
            total_loss_val /= config["NUM_UPDATES_PER_BATCH"]
            total_ratio /= config["NUM_UPDATES_PER_BATCH"]
            total_G0 /= config["NUM_UPDATES_PER_BATCH"]



            # (total_loss, (loss_val, mean_ratio, mean_G0)), grads = \
            #     jax.value_and_grad(loss_fn, has_aux=True)(train_state.params, batch)
            # train_state = train_state.apply_gradients(grads=grads)

            episode_returns = jnp.sum(traj_batch["rewards"] * masks, axis=1)
            mean_return = jnp.mean(episode_returns)
            mean_length = jnp.mean(traj_batch["length"])

            if config.get("DEBUG"):
                def debug_callback(info):
                    print(f"Loss: {info['loss']:.3f}, Mean Return: {info['mean_return']:.3f}, Mean Length: {info['mean_length']:.1f}")
                    print(f"Mean Ratio: {info['mean_ratio']:.3f}, Mean G0: {info['mean_G0']:.3f}")
                jax.debug.callback(debug_callback, 
                                   {
                    "loss": total_loss_val,
                    "mean_return": mean_return,
                    "mean_length": mean_length,
                    "mean_ratio": total_ratio,
                    "mean_G0": total_G0,
                    
                                   }
    
                
                )

            rng, rng_reset = jax.random.split(rng)
            reset_rngs = jax.random.split(rng_reset, config["NUM_ENVS"])
            new_obsv, new_env_state = jax.vmap(env.reset, in_axes=(0, None))(
                reset_rngs, env_params
            )
            new_runner_state = (train_state, new_env_state, new_obsv, rng)
            return new_runner_state, mean_return

        runner_state = (train_state, env_state, obsv, rng)
        runner_state, returns = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "returns": returns}

    return train


if __name__ == "__main__":
    import jax
    # jax.config.update("jax_disable_jit", True)
    # jax.config.update('jax_platform_name', 'cpu')
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 32,
        "TOTAL_EPISODES": 500,
        "GAMMA": 1.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "relu",
        "ENV_NAME": "Acrobot-v1",
        # "ENV_NAME": "Asterix-MinAtar",
        "MAX_EPISODE_LENGTH": 500,
        "NUM_UPDATES_PER_BATCH": 20,
        "delta": 1.0,
        "epsilon": 0.05,
        "tau": 1e-3,
        "alpha_dual": 1e-2,
        "tol": 1e-4,
        "max_dual_iters": 200,
        "damping": 1e-1,
        "DEBUG": True,
    }
    # rng = jax.random.PRNGKey(30)
    # train_jit = jax.jit(make_train(config))
    # out = train_jit(rng)
    rng = jax.random.PRNGKey(30)
    rngs = jax.random.split(rng, 30)
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit, in_axes=(0))(rngs)
    print(out["returns"].mean(axis=0))
    import matplotlib.pyplot as plt
    plt.plot(out["returns"].mean(axis=0))
    plt.savefig("reinforce_trpo.png")