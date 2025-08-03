import numpy as np
import tensorflow as tf
from collections import deque
import random
from tqdm import trange

policy = tf.keras.mixed_precision.Policy('float32')
tf.keras.mixed_precision.set_global_policy(policy)

class FixedNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None
        self._initialized = False

    def fit(self, X):
        X = np.array(X)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        self._initialized = True

    def normalize(self, X):
        if not self._initialized:
            raise ValueError("Normalizer not fitted yet.")
        return np.clip((X - self.mean) / self.std, -5, 5)

class OUNoise:
    def __init__(self, size, mu=0., theta=0.15, sigma=0.3):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.stack, zip(*batch))

    def __len__(self):
        return len(self.buffer)

class ActorCriticAgent:
    def __init__(self, env, gamma=0.99, tau=0.005, actor_lr=1e-4, critic_lr=3e-4,
                 buffer_size=100000, batch_size=128, warm_up=5000):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.gamma, self.tau = gamma, tau
        self.batch_size, self.warm_up = batch_size, warm_up

        sample_states = []
        for _ in range(100):
            state, _ = env.reset()
            sample_states.append(state.flatten())
        self.normalizer = FixedNormalizer()
        self.normalizer.fit(np.array(sample_states))

        self.actor = self._build_actor()
        self.critic1 = self._build_critic()
        self.critic2 = self._build_critic()
        self.target_actor = self._build_actor()
        self.target_critic1 = self._build_critic()
        self.target_critic2 = self._build_critic()

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(critic_lr)

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.noise = OUNoise(self.action_size)
        self.noise_decay = 0.999

    def _build_actor(self):
        x = inp = tf.keras.Input(shape=(self.state_size,))
        for units in [128, 128, 64]:
            x = tf.keras.layers.Dense(units, activation='relu')(x)
            x = tf.keras.layers.LayerNormalization()(x)
        out = tf.keras.layers.Dense(self.action_size, activation='tanh')(x)
        return tf.keras.Model(inp, out)

    def _build_critic(self):
        s = tf.keras.Input(shape=(self.state_size,))
        a = tf.keras.Input(shape=(self.action_size,))
        sh = tf.keras.layers.Dense(256, activation='relu')(s)
        ah = tf.keras.layers.Dense(256, activation='relu')(a)
        x = tf.keras.layers.Concatenate()([sh, ah])
        for units in [128, 64]:
            x = tf.keras.layers.Dense(units, activation='relu')(x)
            x = tf.keras.layers.LayerNormalization()(x)
        out = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model([s, a], out)

    def act(self, state, add_noise=True):
        state = self.normalizer.normalize(state.reshape(1, -1))
        action = self.actor(state).numpy()[0]
        return np.clip(action + self.noise.sample(), -1, 1) if add_noise else action

    def remember(self, s, a, r, s_, d):
        self.replay_buffer.push(s, a, r, s_, d)

    def _train_critic(self, s, a, r, s_, d):
        norm = lambda x: tf.clip_by_value((x - self.normalizer.mean) / self.normalizer.std, -5, 5)
        a_ = self.target_actor(norm(s_))
        noise = tf.random.normal(tf.shape(a_), 0.0, 0.2)
        a_ = tf.clip_by_value(a_ + tf.clip_by_value(noise, -0.5, 0.5), -1, 1)
        q1_t = self.target_critic1([norm(s_), a_])
        q2_t = self.target_critic2([norm(s_), a_])
        y = r + self.gamma * tf.minimum(q1_t, q2_t) * (1 - d)

        with tf.GradientTape() as t1:
            q1 = self.critic1([norm(s), a])
            l1 = tf.reduce_mean(tf.square(y - q1))
        self.critic1_optimizer.apply_gradients(zip(t1.gradient(l1, self.critic1.trainable_variables), self.critic1.trainable_variables))

        with tf.GradientTape() as t2:
            q2 = self.critic2([norm(s), a])
            l2 = tf.reduce_mean(tf.square(y - q2))
        self.critic2_optimizer.apply_gradients(zip(t2.gradient(l2, self.critic2.trainable_variables), self.critic2.trainable_variables))
        return l1 + l2

    def _train_actor(self, s):
        norm = lambda x: tf.clip_by_value((x - self.normalizer.mean) / self.normalizer.std, -5, 5)
        with tf.GradientTape() as t:
            a = self.actor(norm(s))
            loss = -tf.reduce_mean(self.critic1([norm(s), a]))
        self.actor_optimizer.apply_gradients(zip(t.gradient(loss, self.actor.trainable_variables), self.actor.trainable_variables))
        return loss

    def update_targets(self):
        soft_update = lambda t, s: self.tau * s + (1 - self.tau) * t
        self.target_actor.set_weights([soft_update(t, s) for s, t in zip(self.actor.get_weights(), self.target_actor.get_weights())])
        self.target_critic1.set_weights([soft_update(t, s) for s, t in zip(self.critic1.get_weights(), self.target_critic1.get_weights())])
        self.target_critic2.set_weights([soft_update(t, s) for s, t in zip(self.critic2.get_weights(), self.target_critic2.get_weights())])

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None
        s, a, r, s_, d = self.replay_buffer.sample(self.batch_size)
        s, a, r, s_, d = map(lambda x: tf.convert_to_tensor(x, dtype=tf.float32), [s, a, r.reshape(-1, 1), s_, d.reshape(-1, 1)])
        critic_loss = self._train_critic(s, a, r, s_, d)
        actor_loss = None
        if len(self.replay_buffer) % 2 == 0:
            actor_loss = self._train_actor(s)
            self.update_targets()
        return critic_loss, actor_loss

    def train(self, episodes=50, print_every=5):
        net_worths, returns = [], []
        loop = trange(episodes, desc="Training")
        updates_per_episode = 10
        for ep in loop:
            s, info = self.env.reset()
            done, total_r, steps = False, 0, 0
            self.noise.reset()
            updates = 0
            while not done:
                a = self.act(s)
                s_, r, term, trunc, info = self.env.step(a)
                done = term or trunc
                self.remember(s, a, r, s_, float(done))
                if len(self.replay_buffer) >= self.warm_up and updates < updates_per_episode:
                    self.replay()
                    updates += 1
                s, total_r, steps = s_, total_r + r, steps + 1
            nw = info.get("net_worth", self.env.initial_balance)
            pct = ((nw - self.env.initial_balance) / self.env.initial_balance) * 100
            net_worths.append(nw)
            returns.append(pct)
            self.noise.sigma *= self.noise_decay
            if (ep + 1) % print_every == 0:
                loop.set_postfix({"Ep": ep + 1, "Return": f"{pct:.2f}%", "Net": f"${nw:,.0f}", "Steps": steps})
        return net_worths, returns
