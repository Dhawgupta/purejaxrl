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
        # Actor network
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic network
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


def make_train(config):
    # For REINFORCE we treat each full episode as one “trajectory.”
    # Here we define TOTAL_EPISODES to be the number of update iterations.
    config["NUM_UPDATES"] = config["TOTAL_EPISODES"]
    env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def train(rng):
        # ---------------------------
        # INIT NETWORK & OPTIMIZER
        # ---------------------------
        network = ActorCritic(
            env.action_space(env_params).n, activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        tx = optax.chain(
            # optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.rmsprop(config["LR"], eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply, params=network_params, tx=tx
        )

        # ---------------------------
        # INIT ENVIRONMENTS
        # ---------------------------
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        max_ep_len = config.get("MAX_EPISODE_LENGTH", 500)

        # ------------------------------------------------------------
        # RUN A SINGLE EPISODE (TRAJECTORY) UNTIL TERMINATION OR TIMEOUT
        # ------------------------------------------------------------
        def run_episode(train_state, env_state, obs, rng):
            # Preallocate fixed buffers for a full episode.
            obs_buffer = jnp.zeros((max_ep_len,) + obs.shape)
            actions_buffer = jnp.zeros((max_ep_len,), dtype=jnp.int32)
            rewards_buffer = jnp.zeros((max_ep_len,))
            log_probs_buffer = jnp.zeros((max_ep_len,))
            values_buffer = jnp.zeros((max_ep_len,))

            # We use a loop counter t and stop when either t == max_ep_len or done becomes True.
            def cond_fun(state):
                t, env_state, obs, rng, done, *_ = state
                return jnp.logical_and(t < max_ep_len, jnp.logical_not(done))

            def body_fun(state):
                t, env_state, obs, rng, done, obs_buffer, actions_buffer, rewards_buffer, log_probs_buffer, values_buffer = state
                rng, rng_action = jax.random.split(rng)
                pi, value = network.apply(train_state.params, obs)
                action = pi.sample(seed=rng_action)
                log_prob = pi.log_prob(action)

                # Store current step data.
                obs_buffer = obs_buffer.at[t].set(obs)
                actions_buffer = actions_buffer.at[t].set(action)
                log_probs_buffer = log_probs_buffer.at[t].set(log_prob)
                values_buffer = values_buffer.at[t].set(value)

                # Step the environment.
                rng, rng_step = jax.random.split(rng)
                obs_next, env_state, reward, done, _ = env.step(
                    rng_step, env_state, action, env_params
                )
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

            # Return the full buffers (of shape [max_ep_len, ...]) and the actual trajectory length.
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
                obs,  # last observation (not used here, but could be)
                rng,
            )

        # ------------------------------------------------------------
        # Compute Discounted Returns for a Single Trajectory
        # ------------------------------------------------------------
        def compute_returns_old(rewards, length):
            # Compute G_t = r_t + gamma * G_{t+1} for valid steps only.
            def body_fun(carry, r):
                return carry * config["GAMMA"] + r, carry * config["GAMMA"] + r

            rewards_valid = rewards[:length]
            _, returns_valid = jax.lax.scan(body_fun, 0.0, rewards_valid[::-1])
            returns_valid = returns_valid[::-1]
            pad_size = max_ep_len - length
            returns = jnp.concatenate([returns_valid, jnp.zeros(pad_size)])
            return returns
    
        def compute_returns(rewards, mask):
            # rewards: shape (max_ep_len,), mask: shape (max_ep_len,) with 1 for valid steps and 0 for padded steps.
            def body_fun(carry, r_m):
                r, m = r_m
                new_carry = r + config["GAMMA"] * carry * m
                return new_carry, new_carry
            # Process in reverse order and then flip back
            _, returns = jax.lax.scan(body_fun, 0.0, (rewards[::-1], mask[::-1]))
            returns = returns[::-1]
            return returns


        # Vectorize return computation over a batch (one trajectory per environment).
        v_compute_returns = jax.vmap(compute_returns, in_axes=(0, 0))

        # ------------------------------------------------------------
        # UPDATE STEP: Collect one trajectory per env, compute loss & update.
        # ------------------------------------------------------------
        

        def _update_step(runner_state, unused):
            train_state, env_state, obsv, rng = runner_state
            rng, rng_epi = jax.random.split(rng)
            epi_rngs = jax.random.split(rng_epi, config["NUM_ENVS"])
            # Run episodes in parallel across NUM_ENVS.
            results = jax.vmap(run_episode, in_axes=(None, 0, 0, 0))(
                train_state, env_state, obsv, epi_rngs
            )
            traj_batch, new_env_state, new_obsv, rngs = results
            # traj_batch is a dict with keys: "obs", "actions", "rewards", "log_probs", "values", "length"
            lengths = traj_batch["length"]  # shape: (NUM_ENVS,)

            # Create a mask for valid timesteps in each trajectory.
            timesteps = jnp.arange(max_ep_len)[None, :]  # shape (1, max_ep_len)
            masks = timesteps < lengths[:, None]           # shape (NUM_ENVS, max_ep_len)

            # Compute discounted returns.
            returns = v_compute_returns(traj_batch["rewards"], masks)
            # For a baseline version, advantages could be returns minus the stored values,
            # but here we recompute the loss so that gradients flow properly.
            advantages = returns - traj_batch["values"]

            # Flatten the batch (across envs and timesteps) and apply the mask.
            flat_mask = masks.flatten()
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_obs = traj_batch["obs"].reshape(-1, *traj_batch["obs"].shape[2:])
            flat_actions = traj_batch["actions"].reshape(-1)

            # Prepare a batch dictionary.
            batch = {
                "flat_obs": flat_obs,
                "flat_actions": flat_actions,
                "flat_advantages": flat_advantages,
                "flat_returns": flat_returns,
                "flat_mask": flat_mask,
            }

            # Define an explicit loss function.
            def loss_fn(params, batch):
                # Re-run the network to obtain current outputs.
                pi, current_values = network.apply(params, batch["flat_obs"])
                current_log_probs = pi.log_prob(batch["flat_actions"])
                total_mask = jnp.sum(batch["flat_mask"])
                # Compute policy loss (using the freshly computed log probs).
                policy_loss = -jnp.sum(current_log_probs * batch["flat_advantages"] * batch["flat_mask"]) / total_mask
                # Compute value loss.
                value_loss = jnp.sum(jnp.square(batch["flat_returns"] - current_values) * batch["flat_mask"]) / total_mask
                # Compute entropy bonus.
                entropy_loss = -jnp.sum(pi.entropy() * batch["flat_mask"]) / total_mask
                total_loss = policy_loss + config["VF_COEF"] * value_loss + config.get("ENT_COEF", 0.0) * entropy_loss
                return total_loss, (policy_loss, value_loss, entropy_loss)

            # Compute loss and gradients.
            (total_loss, (policy_loss, value_loss, entropy_loss)), grads = \
                jax.value_and_grad(loss_fn, has_aux=True)(train_state.params, batch)
            train_state = train_state.apply_gradients(grads=grads)

            # Compute additional metrics for logging.
            episode_returns = jnp.sum(traj_batch["rewards"] * masks, axis=1)
            mean_return = jnp.mean(episode_returns)
            mean_length = jnp.mean(traj_batch["length"])

            # Debug callback to print training statistics.
            if config.get("DEBUG"):
                def debug_callback(info):
                    print(f"Loss: {info['loss']:.3f}, Mean Return: {info['mean_return']:.3f}, Mean Length: {info['mean_length']:.1f}")
                    print(f"Policy Loss: {info['policy_loss']:.3f}, Value Loss: {info['value_loss']:.3f}, Entropy Loss: {info['entropy_loss']:.3f}")
                    print(f"Valid Count: {info['valid_count']}")
                jax.debug.callback(debug_callback, {
                    "loss": total_loss,
                    "mean_return": mean_return,
                    "mean_length": mean_length,
                    "valid_count": jnp.sum(flat_mask),
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy_loss": entropy_loss,
                })

            # Reset the environments for the next update step.
            rng, rng_reset = jax.random.split(rng)
            reset_rngs = jax.random.split(rng_reset, config["NUM_ENVS"])
            new_obsv, new_env_state = jax.vmap(env.reset, in_axes=(0, None))(
                reset_rngs, env_params
            )
            new_runner_state = (train_state, new_env_state, new_obsv, rng)
            return new_runner_state, total_loss



        def _update_step_old(runner_state, unused):
            train_state, env_state, obsv, rng = runner_state
            rng, rng_epi = jax.random.split(rng)
            epi_rngs = jax.random.split(rng_epi, config["NUM_ENVS"])
            # Run episodes in parallel over the NUM_ENVS.
            results = jax.vmap(run_episode, in_axes=(None, 0, 0, 0))(
                train_state, env_state, obsv, epi_rngs
            )
            traj_batch, new_env_state, new_obsv, rngs = results
            # traj_batch is a dict with keys: obs, actions, rewards, log_probs, values, length.
            lengths = traj_batch["length"]  # shape: (NUM_ENVS,)

            # Create a mask for valid timesteps in each trajectory.
            timesteps = jnp.arange(max_ep_len)[None, :]  # shape (1, max_ep_len)
            masks = timesteps < lengths[:, None]  # shape (NUM_ENVS, max_ep_len)

            # Compute discounted returns and then advantages (return - baseline).
            # v_compute_returns = jax.vmap(compute_returns, in_axes=(0, 0))
            returns = v_compute_returns(traj_batch["rewards"], masks)
            # centered_advantages = returns - (jnp.sum(returns * masks) / jnp.sum(masks))
            # returns = v_compute_returns(traj_batch["rewards"], lengths)
            advantages = returns - traj_batch["values"]
            # vanilla reinfore
            # advantages = advantages -  (jnp.sum(advantages * masks) / jnp.sum(masks))
            # advantages = centered_advantages

            # Flatten the batch (across envs and timesteps) and use the mask.
            flat_mask = masks.flatten()
            # flat_log_probs = traj_batch["log_probs"].reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_values = traj_batch["values"].reshape(-1)
            flat_returns = returns.reshape(-1)

            # To compute an entropy bonus, re-run the network on the observations.
            flat_obs = traj_batch["obs"].reshape(-1, *traj_batch["obs"].shape[2:])
            pi, current_values = network.apply(train_state.params, flat_obs)
            flat_log_probs = pi.log_prob(traj_batch["actions"].reshape(-1)) * flat_mask
            # flat_log_probs = current_log_probs.reshape(-1) * flat_mask
            entropy = pi.entropy()

            valid_count = jnp.sum(flat_mask)
            policy_loss = -jnp.sum(flat_log_probs * flat_advantages * flat_mask) / valid_count
            value_loss = jnp.sum(jnp.square(flat_returns - flat_values) * flat_mask) / valid_count
            entropy_loss = -jnp.sum(entropy * flat_mask) / valid_count

            total_loss = policy_loss  + config["VF_COEF"] * value_loss
            # + config["ENT_COEF"] * entropy_loss

                # Compute additional metrics: mean episode return and mean episode length.
            episode_returns = jnp.sum(traj_batch["rewards"] * masks, axis=1)
            mean_return = jnp.mean(episode_returns)
            mean_length = jnp.mean(traj_batch["length"])
            # Compute gradients and update the network parameters.
            grads = jax.grad(lambda params: total_loss)(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
             # Print training stats if DEBUG is enabled.
            if config.get("DEBUG"):
                def debug_callback(info):
                    print(f"Loss: {info['loss']:.3f}, Mean Return: {info['mean_return']:.3f}, Mean Length: {info['mean_length']:.1f}")
                    print(f"Policy Loss: {info['policy_loss']:.3f}, Value Loss: {info['value_loss']:.3f}, Entropy Loss: {info['entropy_loss']:.3f}")
                    print(f"Valid Count: {info['valid_count']}")

                    # print(f"Loss: {loss:.3f}")
                jax.debug.callback(debug_callback, {
                    "loss": total_loss,
                    "mean_return": mean_return,
                    "mean_length": mean_length,
                    "valid_count": valid_count,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy_loss": entropy_loss,
                })


            # Reset the environments for the next update step.
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
        # "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5, 
        "ACTIVATION": "relu",
        "ENV_NAME": "Acrobot-v1",
        "MAX_EPISODE_LENGTH": 500,
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
    with open("reinforce.pkl", "wb") as f:
        pickle.dump(out["returns"], f)
    import matplotlib.pyplot as plt
    plt.plot(out["returns"].mean(axis=0))
    plt.savefig("reinforce.png")
