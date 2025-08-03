# actor_critic_agent.py

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random
from collections import deque

class Normalizer:
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.mean_diff_sq = 0

    def update(self, x):
        x = np.asarray(x)
        n_new = x.shape[0]
        delta = x.mean() - self.mean
        self.mean += delta * n_new / (self.n + n_new)
        self.mean_diff_sq += ((x - x.mean())**2).sum() + delta**2 * self.n * n_new / (self.n + n_new)
        self.n += n_new

    def normalize(self, x):
        std = np.sqrt(self.mean_diff_sq / self.n) if self.n > 0 else 1
        return (x - self.mean) / (std + 1e-8)


class OUNoise:
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class Actor(tf.keras.Model):
    def __init__(self, action_dim):
        super().__init__()
        self.base = tf.keras.Sequential([
            layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
            layers.LayerNormalization(),
            layers.Dense(64, activation="relu"),
            layers.Dense(action_dim, activation="tanh")
        ])

    def call(self, state):
        return self.base(state)


class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.base = tf.keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.LayerNormalization(),
            layers.Dense(64, activation="relu"),
            layers.Dense(1)
        ])

    def call(self, state, action):
        x = tf.concat([state, action], axis=1)
        return self.base(x)


class ActorCriticAgent:
    def __init__(
        self,
        env,
        gamma=0.99,
        tau=0.005,
        actor_lr=1e-4,
        critic_lr=2e-4,
        buffer_size=10000,
        batch_size=64
    ):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

        state, _ = self.env.reset()
        self.obs_dim = state.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        self.actor = Actor(self.act_dim)
        self.critic = Critic()
        self.target_actor = Actor(self.act_dim)
        self.target_critic = Critic()

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

        self.normalizer = Normalizer()
        self.noise = OUNoise(mu=np.zeros(self.act_dim))

    def normalize_obs(self, obs):
        flat = np.array(obs).flatten().astype(np.float32)
        self.normalizer.update(flat)
        return self.normalizer.normalize(flat)

    def get_action(self, state):
        flat_state = self.normalize_obs(state)
        state_tensor = tf.convert_to_tensor([flat_state], dtype=tf.float32)
        action = self.actor(state_tensor)[0].numpy()
        noise = self.noise()
        action = np.clip(action + noise, -1, 1)
        return action

    def reset(self):
        self.noise.reset()

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((self.normalize_obs(state), action, reward, self.normalize_obs(next_state), float(done)))

    def sample_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def soft_update(self, source_model, target_model):
        for target_var, source_var in zip(target_model.trainable_variables, source_model.trainable_variables):
            target_var.assign(self.tau * source_var + (1.0 - self.tau) * target_var)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_batch()
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            next_actions = self.target_actor(next_states)
            target_q = rewards + (1.0 - dones) * self.gamma * tf.squeeze(self.target_critic(next_states, next_actions))
            target_q = tf.stop_gradient(target_q)

            current_q = tf.squeeze(self.critic(states, actions))
            critic_loss = tf.reduce_mean(tf.square(target_q - current_q))

            actor_loss = -tf.reduce_mean(self.critic(states, self.actor(states)))

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

    def train(self, episodes=10):
        all_net_worths = []
        episode_net_worth = []

        for ep in range(episodes):
            state, _ = self.env.reset()
            self.reset()
            done = False
            total_reward = 0
            episode_trades = []

            while not done:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                reward = np.clip(reward, -0.05, 0.05)
                self.store(state, action, reward, next_state, done)
                self.train_step()
                state = next_state
                total_reward += reward

                if done:
                    net_worth = info["net_worth"]
                    if isinstance(net_worth, np.ndarray):
                        net_worth = net_worth.item()

                    trades = info["trades"]
                    episode_trades.extend(trades)

                    nw_history = info.get("net_worths", [])
                    if not isinstance(nw_history, list):
                        nw_history = [float(net_worth)]

                    all_net_worths.append(nw_history)
                    episode_net_worth.append(float(net_worth))

                    print(f"Episode: {ep}, Total Reward: {total_reward:.4f}, Final Net Worth: {net_worth:.2f}")

        return all_net_worths, episode_net_worth