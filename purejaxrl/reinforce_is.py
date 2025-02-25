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

        # Critic network (not used in IS objective, but computed).
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

        # return pi, jnp.squeeze(critic, axis=-1)
        return pi, 0.0


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
            optax.rmsprop(config["LR"], eps=1e-5),
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

            # Define the explicit loss function for the IS objective.
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
            for _ in range(config["NUM_UPDATES_PER_BATCH"]):
                (loss_val_, (loss_val, mean_ratio, mean_G0)), grads = \
                    jax.value_and_grad(loss_fn, has_aux=True)(train_state.params, batch)
                total_loss += loss_val_
                total_loss_val += loss_val
                total_ratio += mean_ratio
                total_G0 += mean_G0
                train_state = train_state.apply_gradients(grads=grads)


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
                #                    {
                #     "loss": loss_val,
                #     "mean_return": mean_return,
                #     "mean_length": mean_length,
                #     "mean_ratio": mean_ratio,
                #     "mean_G0": mean_G0,
                # }
                
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
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 32,
        "TOTAL_EPISODES": 300,  # Total update iterations (each based on NUM_ENVS full episodes)
        "GAMMA": 1.0,
        "ACTIVATION": "relu",
        "ENV_NAME": "Acrobot-v1",
        "MAX_EPISODE_LENGTH": 500,
        "NUM_UPDATES_PER_BATCH": 20,
        "DEBUG": False,
    }
    rng = jax.random.PRNGKey(30)
    rngs = jax.random.split(rng, 60)
    
    train_jit = jax.jit(make_train(config))
    import time
    current_time = time.time()
    out = jax.vmap(train_jit, in_axes=(0))(rngs)
    print("Time taken: ", time.time() - current_time)
    # print(out["returns"].mean(axis=0))
    # save the results
    import pickle
    with open("reinforce_is.pkl", "wb") as f:
        pickle.dump(out["returns"], f)
    import matplotlib.pyplot as plt
    plt.plot(out["returns"].mean(axis=0))
    plt.savefig("reinforce_is.png")