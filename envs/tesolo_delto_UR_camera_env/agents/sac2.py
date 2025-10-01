import os, time, math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== Совместимость: склеиваем dict наблюдений в (N, D)
def obs_to_tensor(obs_dict):
    grp = obs_dict["state"] if isinstance(obs_dict, dict) and "state" in obs_dict else obs_dict
    parts = []
    for _, v in grp.items():
        t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v, device=device)
        parts.append(t.to(device))
    x = torch.cat(parts, dim=-1)
    return x


# ======= Политика: tanh-сквошенная гауссиана с корректным logπ
class TanhGaussianPolicy(nn.Module):
    """
    Непрерывная политика N(μ, σ) с tanh-сквошем:
      u ~ N(μ, σ), a = tanh(u) \in (-1, 1)^A
    Корректируем logπ: logπ(a|s) = logN(u|μ,σ) - sum log(1 - tanh(u)^2)
    """
    def __init__(self, obs_dim, act_dim, hidden=256, log_std_bounds=(-5.0, 2.0)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.log_std_min, self.log_std_max = log_std_bounds

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return mu, std

    def sample(self, obs):
        mu, std = self(obs)
        dist = Normal(mu, std)
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        # jacobian correction for tanh
        logp = dist.log_prob(u).sum(-1) - torch.log1p(-a.pow(2) + 1e-6).sum(-1)
        return a, logp, torch.tanh(mu)

    @torch.no_grad()
    def act_mean(self, obs):
        mu, _ = self(obs)
        return torch.tanh(mu)


class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = int(capacity)
        self.ptr = 0
        self.full = False
        self.obs = torch.zeros(self.capacity, obs_dim, device=device)
        self.act = torch.zeros(self.capacity, act_dim, device=device)
        self.rew = torch.zeros(self.capacity, device=device)
        self.next = torch.zeros(self.capacity, obs_dim, device=device)
        self.done = torch.zeros(self.capacity, device=device)  # здесь храним ИМЕННО terminated (для бутстрапа)

    def add(self, obs, act, rew, next_obs, done_bootstrap):
        n = obs.shape[0]
        idxs = (torch.arange(n, device=device) + self.ptr) % self.capacity
        self.obs[idxs] = obs
        self.act[idxs] = act
        self.rew[idxs] = rew
        self.next[idxs] = next_obs
        self.done[idxs] = done_bootstrap
        self.ptr = int((self.ptr + n) % self.capacity)
        if self.ptr == 0:
            self.full = True

    def size(self):
        return self.capacity if self.full else self.ptr

    def sample(self, batch_size: int):
        max_n = self.size()
        idx = torch.randint(0, max_n, (batch_size,), device=device)
        return (self.obs[idx], self.act[idx], self.rew[idx], self.next[idx], self.done[idx])


@dataclass
class Checkpoint:
    actor: nn.Module
    q1: nn.Module
    q2: nn.Module
    q1_targ: nn.Module
    q2_targ: nn.Module
    opt_actor: torch.optim.Optimizer
    opt_q: torch.optim.Optimizer
    log_alpha: torch.Tensor
    opt_alpha: torch.optim.Optimizer
    step_counter: int = 0

    def save(self, path: str, extra: dict | None = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_targ": self.q1_targ.state_dict(),
            "q2_targ": self.q2_targ.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_q": self.opt_q.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "opt_alpha": self.opt_alpha.state_dict(),
            "steps": self.step_counter,
            "ts": time.time(),
        }
        if extra: payload["extra"] = extra
        torch.save(payload, path)

    def load(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location or device)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_targ.load_state_dict(ckpt["q1_targ"])
        self.q2_targ.load_state_dict(ckpt["q2_targ"])
        self.opt_actor.load_state_dict(ckpt["opt_actor"])
        self.opt_q.load_state_dict(ckpt["opt_q"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"].to(device))
        self.opt_alpha.load_state_dict(ckpt["opt_alpha"])
        self.step_counter = int(ckpt.get("steps", 0))
        return ckpt.get("extra", None)


class SAC:
    def __init__(
        self, env, obs_dim, act_dim,
        actor_hidden=256, critic_hidden=256,
        gamma=0.98, tau=0.01,
        lr_actor=3e-4, lr_critic=1e-4, lr_alpha=3e-4,
        target_entropy=None,
        buffer_capacity=500_000,
        batch_size=256,
        updates_per_step=1,
        start_random_steps=2048,
        save_dir="runs/hand_sac"
    ):
        self.env = env
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.gamma, self.tau = gamma, tau
        self.batch_size = batch_size
        self.updates_per_step = updates_per_step
        self.start_random_steps = start_random_steps
        self.num_envs = getattr(env, "num_envs", 1)
        self.save_dir = save_dir

        # сети
        self.actor = TanhGaussianPolicy(obs_dim, act_dim, hidden=actor_hidden).to(device)
        self.q1 = QNet(obs_dim, act_dim, hidden=critic_hidden).to(device)
        self.q2 = QNet(obs_dim, act_dim, hidden=critic_hidden).to(device)
        self.q1_targ = QNet(obs_dim, act_dim, hidden=critic_hidden).to(device)
        self.q2_targ = QNet(obs_dim, act_dim, hidden=critic_hidden).to(device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        # оптимизаторы
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr_critic)

        # температура (энтропия)
        self.target_entropy = float(-(act_dim) if target_entropy is None else target_entropy)
        self.log_alpha = torch.tensor(math.log(0.2), device=device, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        # буфер
        self.replay = ReplayBuffer(buffer_capacity, obs_dim, act_dim)

        # чекпоинт
        self.ckpt = Checkpoint(
            self.actor, self.q1, self.q2, self.q1_targ, self.q2_targ,
            self.opt_actor, self.opt_q, self.log_alpha, self.opt_alpha, 0
        )

        self.last_log = {}
        self.best_return = -1e9

        self.rewards = []
        self.q_losses = []
        self.pi_losses = []

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def _select_action(self, obs, deterministic=False):
        if deterministic:
            a = self.actor.act_mean(obs)
            a = torch.clamp(a, -1.0, 1.0)
            logp = None
        else:
            a, logp, _ = self.actor.sample(obs)
            a = torch.clamp(a, -1.0, 1.0)
        return a, logp

    def _update(self):
        if self.replay.size() < self.batch_size:
            return

        for _ in range(self.updates_per_step):
            obs, act, rew, next_obs, done_bootstrap = self.replay.sample(self.batch_size)

            # ----- target для Q
            with torch.no_grad():
                a2, logp2, _ = self.actor.sample(next_obs)
                a2 = torch.clamp(a2, -1.0, 1.0)
                q1_t = self.q1_targ(next_obs, a2)
                q2_t = self.q2_targ(next_obs, a2)
                q_targ_min = torch.min(q1_t, q2_t)
                y = rew + (1.0 - done_bootstrap) * self.gamma * (q_targ_min - self.alpha * logp2)

            # ----- обновление критиков (Huber + clip grad)
            q1_pred = self.q1(obs, act)
            q2_pred = self.q2(obs, act)
            q_loss = F.smooth_l1_loss(q1_pred, y) + F.smooth_l1_loss(q2_pred, y)

            self.opt_q.zero_grad(set_to_none=True)
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)
            self.opt_q.step()

            # ----- обновление актора
            a_new, logp_new, _ = self.actor.sample(obs)
            a_new = torch.clamp(a_new, -1.0, 1.0)
            q1_pi = self.q1(obs, a_new)
            q2_pi = self.q2(obs, a_new)
            q_pi_min = torch.min(q1_pi, q2_pi)
            pi_loss = (self.alpha.detach() * logp_new - q_pi_min).mean()

            self.opt_actor.zero_grad(set_to_none=True)
            pi_loss.backward()
            self.opt_actor.step()

            # ----- авто-тюнинг температуры
            alpha_loss = -(self.log_alpha * (logp_new.detach() + self.target_entropy)).mean()
            self.opt_alpha.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.opt_alpha.step()

            # ----- Polyak-обновление таргетов
            with torch.no_grad():
                for p, pt in zip(self.q1.parameters(), self.q1_targ.parameters()):
                    pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
                for p, pt in zip(self.q2.parameters(), self.q2_targ.parameters()):
                    pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        # метрики
        self.last_log.update({
            "q_loss": float(q_loss.item()),
            "pi_loss": float(pi_loss.item()),
            "alpha": float(self.alpha.item()),
        })

    @torch.no_grad()
    def evaluate(self, env, episodes: int = 5, render: bool = False, max_steps: int | None = None):
        # Безопасный лимит шага
        if max_steps is None:
            if hasattr(env, "max_episode_length") and isinstance(env.max_episode_length, int):
                max_steps = env.max_episode_length
            else:
                max_steps = 2000

        returns = []
        for _ in range(episodes):
            obs_dict = env.reset()[0]
            obs = obs_to_tensor(obs_dict)
            ep_ret = torch.zeros(env.num_envs, device=device)
            done_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

            for t in range(max_steps):
                act = self.actor.act_mean(obs)
                act = torch.clamp(act, -1.0, 1.0)
                next_obs_dict, rew, dones, info, _ = env.step(act)
                ep_ret += rew * (~done_mask)
                done_mask |= dones
                obs = obs_to_tensor(next_obs_dict)
                if done_mask.all(): break
                if render and hasattr(env, "render"): env.render()

            returns.append(ep_ret.mean().item())

        return float(sum(returns) / len(returns))

    def train(self, total_env_steps: int, log_interval_updates: int = 10):
        obs_dict = self.env.reset()[0]
        obs = obs_to_tensor(obs_dict)

        updates = 0
        wall_t0 = time.time()
        prew_t = time.time()
        rollout_rew_accum = 0.0
        rollout_step_accum = 0
        counter = 0

        while self.ckpt.step_counter < total_env_steps:
            # ===== шаг интеракции
            if self.ckpt.step_counter < self.start_random_steps:
                act = torch.empty((self.env.num_envs, self.act_dim), device=device).uniform_(-0.5, 0.5)
                logp = None
            else:
                act, logp = self._select_action(obs, deterministic=False)

            next_obs_dict, rew, dones, info, _ = self.env.step(act)
            next_obs = obs_to_tensor(next_obs_dict)

            # Разделяем истинный терминал и time-limit (если среда сообщает)
            terminated = None
            if isinstance(info, dict):
                if "terminated" in info:
                    terminated = info["terminated"]
                # 'time_outs' или 'truncated' могут быть, но для бутстрапа их НЕ считаем терминалом
            done_bootstrap = (terminated if isinstance(terminated, torch.Tensor) else dones).float()

            # аккумулируем среднюю награду
            rollout_rew_accum += rew.mean().item()
            rollout_step_accum += 1

            # кладём в буфер РОВНО то действие, что пошло в среду
            self.replay.add(obs, act, rew, next_obs, done_bootstrap)

            # шаг вперёд
            obs = next_obs

            # апдейты
            self._update()

            # счётчик шагов (сумма по всем копиям)
            self.ckpt.step_counter += self.env.num_envs

            # ===== лог/чекпоинт
            if self.ckpt.step_counter // self.env.num_envs % log_interval_updates == 0:
                updates += 1
                avg_rollout_rew = rollout_rew_accum / max(1, rollout_step_accum)
                sps = (self.env.num_envs * rollout_step_accum) / max(1e-6, (time.time() - prew_t))
                prew_t = time.time()
                wall_m = (time.time() - wall_t0) / 60.0

                self.rewards.append(avg_rollout_rew)
                self.q_losses.append(self.last_log.get('q_loss', float('nan')))
                self.pi_losses.append(self.last_log.get('pi_loss', float('nan')))

                print(
                    f"[upd {updates:04d}] steps={self.ckpt.step_counter:,}  "
                    f"rew={avg_rollout_rew:.3f}  "
                    f"q_loss={self.last_log.get('q_loss', float('nan')):.4f}  "
                    f"pi_loss={self.last_log.get('pi_loss', float('nan')):.4f}  "
                    f"alpha={self.last_log.get('alpha', float('nan')):.3f}  "
                    f"SPS={sps:.0f}  t={wall_m:.1f}m"
                )
                rollout_rew_accum = 0.0
                rollout_step_accum = 0

                counter += 1

                if counter >= 100:
                    
                    counter = 0
                    avg_eval = self.evaluate(self.env, episodes=2)
                    self.ckpt.save(os.path.join(self.save_dir, "last.pt"), extra={"eval_avg_return": avg_eval})
                    if avg_eval > self.best_return:
                        self.best_return = avg_eval
                        self.ckpt.save(os.path.join(self.save_dir, "best.pt"), extra={"eval_avg_return": avg_eval})
                    print(f"[ckpt] eval_avg_ret={avg_eval:.3f}  best={self.best_return:.3f}  saved→ {self.save_dir}")

                    with open(self.save_dir + "/saves.txt", "w") as output:
                        output.write(str(self.rewards)+"\n")
                        output.write(str(self.q_losses)+"\n")
                        output.write(str(self.pi_losses)+"\n")
                        output.close()


    def load_demo_to_buffer(self, path: str):
        data = np.load(path)
        obs      = torch.tensor(data["obs"], dtype=torch.float32, device=device)
        actions  = torch.tensor(data["actions"], dtype=torch.float32, device=device)
        rewards  = torch.tensor(data["rewards"], dtype=torch.float32, device=device).squeeze(-1)
        next_obs = torch.tensor(data["next_obs"], dtype=torch.float32, device=device)
        dones    = torch.tensor(data["dones"], dtype=torch.float32, device=device).squeeze(-1)

        self.replay.add(obs, actions, rewards, next_obs, dones)
        print(f"[Buffer] Loaded {obs.shape[0]} transitions from demo into replay buffer.")
                    


