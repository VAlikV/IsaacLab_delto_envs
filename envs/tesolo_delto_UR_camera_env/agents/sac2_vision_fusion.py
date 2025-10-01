import os, time, math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# =============================================================
# Device
# =============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================
# Vision config & preprocessing
# =============================================================

@dataclass
class VisionCfg:
    img_size: Tuple[int, int] = (84, 84)   # (H, W)
    frame_stack: int = 1                   # set >1 if you maintain a frame stack outside
    grayscale: bool = False
    normalize01: bool = True               # divide by 255
    aug_random_shift: int = 4              # DrQ-style; 0 disables
    encoder_out_dim: int = 256


def _ensure_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x, device=device)


def preprocess_rgb(rgb_uint8: torch.Tensor, cfg: VisionCfg) -> torch.Tensor:
    """
    rgb_uint8: (B,H,W,3) uint8 or float in [0..255]
    returns: (B,C,H,W) float32 in [0..1] (if normalize01)
    """
    x = rgb_uint8
    if x.dtype != torch.uint8:
        # assume [0..255]
        x = x.clamp(0, 255).to(torch.uint8)
    x = x.float()
    if cfg.normalize01:
        x = x / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
    if cfg.grayscale:
        x = x.mean(dim=1, keepdim=True)
    x = F.interpolate(x, size=cfg.img_size, mode="bilinear", align_corners=False)
    return x


@torch.no_grad()
def random_shift(img: torch.Tensor, pad: int) -> torch.Tensor:
    """DrQ-style random shift. img: (N,C,H,W) -> (N,C,H,W)"""
    if pad <= 0:
        return img
    n, c, h, w = img.shape
    padded = F.pad(img, (pad, pad, pad, pad), mode='replicate')
    offs_h = torch.randint(0, 2*pad+1, (n,), device=img.device)
    offs_w = torch.randint(0, 2*pad+1, (n,), device=img.device)
    out = torch.empty_like(img)
    for i in range(n):
        out[i] = padded[i, :, offs_h[i]:offs_h[i]+h, offs_w[i]:offs_w[i]+w]
    return out


# =============================================================
# Encoders & nets
# =============================================================

class NatureCNN(nn.Module):
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2),          nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1),          nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 512), nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class FusionEncoder(nn.Module):
    """CNN(rgb) + concat(state) -> fused z"""
    def __init__(self, state_dim: int, vcfg: VisionCfg):
        super().__init__()
        C = (1 if vcfg.grayscale else 3) * vcfg.frame_stack
        self.vcfg = vcfg
        self.cnn = NatureCNN(C, vcfg.encoder_out_dim)
        self.state_norm = nn.LayerNorm(state_dim)
        self.fused_dim = vcfg.encoder_out_dim + state_dim

    def forward(self, rgb_nchw: torch.Tensor, state: torch.Tensor, aug_shift: int = 0):
        x = rgb_nchw
        if aug_shift > 0 and self.training:
            x = random_shift(x, aug_shift)
        z_img = self.cnn(x)
        z = torch.cat([z_img, self.state_norm(state)], dim=-1)
        return z


# =============================================================
# Actor & Critics (same interfaces as your original SAC)
# =============================================================

class TanhGaussianPolicy(nn.Module):
    """
    Continuous policy with tanh squashing.
    """
    def __init__(self, obs_dim, act_dim, hidden=256, log_std_bounds=(-5.0, 2.0)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.log_std_bounds = log_std_bounds

    def forward(self, z):
        h = self.net(z)
        mu = self.mu(h)
        log_std = self.log_std(h)
        log_std = torch.clamp(log_std, self.log_std_bounds[0], self.log_std_bounds[1])
        std = torch.exp(log_std)
        return mu, std

    def sample(self, z):
        mu, std = self(z)
        dist = Normal(mu, std)
        u = dist.rsample()                  # reparameterization
        a = torch.tanh(u)
        # logπ correction for tanh
        logp = dist.log_prob(u).sum(-1, keepdim=True) - torch.log(1 - a.pow(2) + 1e-6).sum(-1, keepdim=True)
        return a, logp, dist

    @torch.no_grad()
    def act_mean(self, z):
        mu, _ = self(z)
        return torch.tanh(mu)


class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return self.q(x)


# =============================================================
# Utilities
# =============================================================

class Checkpoint:
    def __init__(self, *objs):
        self.objs = objs
        self.step_counter = 0

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {f"obj_{i}": o.state_dict() for i, o in enumerate(self.objs) if hasattr(o, 'state_dict')}
        payload["step_counter"] = self.step_counter
        if extra is not None:
            payload["extra"] = extra
        torch.save(payload, path)

    def load(self, path: str, strict: bool = True):
        data = torch.load(path, map_location=device)
        for i, o in enumerate(self.objs):
            if hasattr(o, 'load_state_dict') and f"obj_{i}" in data:
                o.load_state_dict(data[f"obj_{i}"], strict=strict)
        self.step_counter = int(data.get("step_counter", 0))


# =============================================================
# Observation parsing
# =============================================================

def parse_obs(obs_dict: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (state, rgb_uint8)
    - state: (N,S) float32
    - rgb_uint8: (N,H,W,3) uint8
    """
    assert isinstance(obs_dict, dict), "env must return dict with keys 'state' and 'rgb'"
    # state can be dict of tensors/np
    grp = obs_dict.get("state", {})
    parts = []
    for _, v in grp.items():
        t = _ensure_tensor(v).to(device)
        parts.append(t)
    state = torch.cat(parts, dim=-1).to(torch.float32) if parts else None
    if state is None:
        raise RuntimeError("parse_obs: empty 'state' group")

    rgb = obs_dict.get("rgb", None)
    if rgb is None:
        raise RuntimeError("parse_obs: key 'rgb' not found")
    rgb_t = _ensure_tensor(rgb).to(device)
    if rgb_t.dtype != torch.uint8:
        # assume [0..255]
        rgb_t = rgb_t.clamp(0, 255).to(torch.uint8)
    return state, rgb_t


# =============================================================
# Replay buffer for (state, rgb)
# =============================================================

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, act_dim: int, vcfg: VisionCfg):
        self.capacity = int(capacity)
        self.ptr = 0
        self.full = False
        self.vcfg = vcfg
        H, W = vcfg.img_size
        # Store already-resized HxW to avoid resize on every sample
        # self.rgb = torch.zeros(self.capacity, H, W, 3, dtype=torch.uint8, device=device)
        # self.next_rgb = torch.zeros_like(self.rgb)
        # self.state = torch.zeros(self.capacity, state_dim, dtype=torch.float32, device=device)
        # self.next_state = torch.zeros_like(self.state)
        # self.act = torch.zeros(self.capacity, act_dim, dtype=torch.float32, device=device)
        # self.rew = torch.zeros(self.capacity, dtype=torch.float32, device=device)
        # self.done = torch.zeros(self.capacity, dtype=torch.float32, device=device)  # terminated for bootstrap

        self.rgb       = torch.zeros(self.capacity, H, W, 3, dtype=torch.uint8, device='cpu')
        self.next_rgb  = torch.zeros_like(self.rgb)
        self.state     = torch.zeros(self.capacity, state_dim, dtype=torch.float32, device='cpu')
        self.next_state= torch.zeros_like(self.state)
        self.act       = torch.zeros(self.capacity, act_dim, dtype=torch.float32, device='cpu')
        self.rew       = torch.zeros(self.capacity, dtype=torch.float32, device='cpu')
        self.done      = torch.zeros(self.capacity, dtype=torch.float32, device='cpu')

    # @torch.no_grad()
    # def add(self, state, rgb_hwc_uint8, act, rew, next_state, next_rgb_hwc_uint8, done_bootstrap):
    #     n = state.shape[0]
    #     # idxs = (torch.arange(n, device=device) + self.ptr) % self.capacity
    #     idxs = (torch.arange(n) + self.ptr) % self.capacity

    #     # states & actions
    #     self.state[idxs] = state
    #     self.act[idxs] = act
    #     self.rew[idxs] = rew.squeeze(-1) if rew.ndim == 2 else rew
    #     self.next_state[idxs] = next_state
    #     self.done[idxs] = done_bootstrap.squeeze(-1) if done_bootstrap.ndim == 2 else done_bootstrap

    #     # resize rgb to (H,W) once on insert (nearest)
    #     def _resize_uint8(nhwc_uint8):
    #         x = nhwc_uint8.float().permute(0,3,1,2) / 255.0
    #         x = F.interpolate(x, size=self.vcfg.img_size, mode="nearest")
    #         return (x.permute(0,2,3,1) * 255.0).to(torch.uint8)

    #     self.rgb[idxs] = _resize_uint8(rgb_hwc_uint8)
    #     self.next_rgb[idxs] = _resize_uint8(next_rgb_hwc_uint8)

    #     self.ptr = int((self.ptr + n) % self.capacity)
    #     if self.ptr == 0:
    #         self.full = True

    @torch.no_grad()
    def add(self, state, rgb_hwc_uint8, act, rew, next_state, next_rgb_hwc_uint8, done_bootstrap):
        # всё на CPU
        state      = state.detach().to('cpu', non_blocking=True)
        act        = act.detach().to('cpu', non_blocking=True)
        rew        = rew.detach().to('cpu', non_blocking=True)
        next_state = next_state.detach().to('cpu', non_blocking=True)
        done_bootstrap = done_bootstrap.detach().to('cpu', non_blocking=True)

        # rgb уже может быть uint8 на GPU — тоже на CPU
        rgb_hwc_uint8      = rgb_hwc_uint8.detach().to('cpu', non_blocking=True)
        next_rgb_hwc_uint8 = next_rgb_hwc_uint8.detach().to('cpu', non_blocking=True)

        n = state.shape[0]
        idxs = (torch.arange(n) + self.ptr) % self.capacity      # CPU long
        idxs = idxs.to(torch.long)                               # на всякий случай

        # states & actions
        self.state[idxs]      = state
        self.act[idxs]        = act
        self.rew[idxs]        = rew.squeeze(-1) if rew.ndim == 2 else rew
        self.next_state[idxs] = next_state
        self.done[idxs]       = done_bootstrap.squeeze(-1) if done_bootstrap.ndim == 2 else done_bootstrap

        # resize rgb до (H,W) один раз при вставке (на CPU)
        def _resize_uint8(nhwc_uint8):
            x = nhwc_uint8.float().permute(0,3,1,2) / 255.0        # NCHW float
            x = F.interpolate(x, size=self.vcfg.img_size, mode="nearest")
            return (x.permute(0,2,3,1) * 255.0).to(torch.uint8)    # NHWC uint8

        self.rgb[idxs]      = _resize_uint8(rgb_hwc_uint8)
        self.next_rgb[idxs] = _resize_uint8(next_rgb_hwc_uint8)

        self.ptr = int((self.ptr + n) % self.capacity)
        if self.ptr == 0:
            self.full = True


    def size(self):
        return self.capacity if self.full else self.ptr

    def sample(self, batch_size: int, vcfg: VisionCfg):
        
        # max_n = self.size()
        # idx = torch.randint(0, max_n, (batch_size,), device=device)
        # s  = self.state[idx]
        # ns = self.next_state[idx]
        # a  = self.act[idx]
        # r  = self.rew[idx]
        # d  = self.done[idx]
        # rgb   = preprocess_rgb(self.rgb[idx], vcfg)
        # n_rgb = preprocess_rgb(self.next_rgb[idx], vcfg)
    
        idx = torch.randint(0, self.size(), (batch_size,))          # CPU

        s   = self.state[idx].to(device, non_blocking=True)
        ns  = self.next_state[idx].to(device, non_blocking=True)
        a   = self.act[idx].to(device, non_blocking=True)
        r   = self.rew[idx].unsqueeze(-1).to(device, non_blocking=True)
        d   = self.done[idx].unsqueeze(-1).to(device, non_blocking=True)

        rgb_cpu   = self.rgb[idx]        # (B,H,W,3) uint8 on CPU
        n_rgb_cpu = self.next_rgb[idx]

        rgb   = preprocess_rgb(rgb_cpu.to(device, non_blocking=True), self.vcfg)
        n_rgb = preprocess_rgb(n_rgb_cpu.to(device, non_blocking=True), self.vcfg)

        return s, rgb, a, r.unsqueeze(-1), ns, n_rgb, d.unsqueeze(-1)



# =============================================================
# SAC with vision fusion
# =============================================================

class SAC:
    def __init__(
        self,
        env,
        state_dim: int,
        act_dim: int,
        vision_cfg: VisionCfg = VisionCfg(),
        actor_hidden=256,
        critic_hidden=256,
        gamma=0.98,
        tau=0.01,
        lr_actor=3e-4,
        lr_critic=1e-4,
        lr_alpha=3e-4,
        target_entropy: Optional[float] = None,
        buffer_capacity=500_000,
        batch_size=256,
        updates_per_step=1,
        start_random_steps=2048,
        save_dir="runs/sac_vision",
    ):
        self.env = env
        self.state_dim, self.act_dim = state_dim, act_dim
        self.vcfg = vision_cfg
        self.gamma, self.tau = gamma, tau
        self.batch_size = batch_size
        self.updates_per_step = updates_per_step
        self.start_random_steps = start_random_steps
        self.num_envs = getattr(env, "num_envs", 1)
        self.save_dir = save_dir

        # encoder & fused dim
        self.fusion_encoder = FusionEncoder(state_dim, self.vcfg).to(device)
        fused_dim = self.fusion_encoder.fused_dim

        # networks
        self.actor = TanhGaussianPolicy(fused_dim, act_dim, hidden=actor_hidden).to(device)
        self.q1 = QNet(fused_dim, act_dim, hidden=critic_hidden).to(device)
        self.q2 = QNet(fused_dim, act_dim, hidden=critic_hidden).to(device)
        self.q1_targ = QNet(fused_dim, act_dim, hidden=critic_hidden).to(device)
        self.q2_targ = QNet(fused_dim, act_dim, hidden=critic_hidden).to(device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        # optimizers (shared encoder learns with both actor & critics)
        self.opt_actor = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.fusion_encoder.parameters()), lr=lr_actor
        )
        self.opt_q = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()) + list(self.fusion_encoder.parameters()), lr=lr_critic
        )

        # temperature
        self.target_entropy = float(-(act_dim) if target_entropy is None else target_entropy)
        self.log_alpha = torch.tensor(math.log(0.2), device=device, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        # buffer
        self.replay = ReplayBuffer(buffer_capacity, state_dim, act_dim, self.vcfg)

        # checkpoint wrapper
        self.ckpt = Checkpoint(
            self.fusion_encoder,            # <— добавили
            self.actor, self.q1, self.q2, self.q1_targ, self.q2_targ,
            self.opt_actor, self.opt_q, self.log_alpha, self.opt_alpha, 0
        )

        self.last_log: Dict[str, float] = {}
        self.best_return = -1e9
        self.rewards = []
        self.q_losses = []
        self.pi_losses = []

    # ------------- alpha
    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ------------- action selection
    @torch.no_grad()
    def _select_action(self, state, rgb_uint8, deterministic=False):
        rgb = preprocess_rgb(rgb_uint8, self.vcfg).to(device)
        z = self.fusion_encoder(rgb, state, aug_shift=0)
        if deterministic:
            a = self.actor.act_mean(z)
            a = torch.clamp(a, -1.0, 1.0)
            return a, None
        else:
            a, logp, _ = self.actor.sample(z)
            a = torch.clamp(a, -1.0, 1.0)
            return a, logp

    # ------------- update
    def _update(self):
        # print(f"[DEBUG] Replay size: {self.replay.size()}, batch size: {self.batch_size}")
        if self.replay.size() < self.batch_size:
            return
        for _ in range(self.updates_per_step):
            s, rgb, a, r, ns, n_rgb, d = self.replay.sample(self.batch_size, self.vcfg)

            # ----- target
            with torch.no_grad():
                z_next = self.fusion_encoder(n_rgb, ns, aug_shift=self.vcfg.aug_random_shift)
                a2, logp2, _ = self.actor.sample(z_next)
                a2 = torch.clamp(a2, -1.0, 1.0)
                q1_t = self.q1_targ(z_next, a2)
                q2_t = self.q2_targ(z_next, a2)
                q_targ = torch.min(q1_t, q2_t) - self.alpha * logp2
                y = r + (1.0 - d) * self.gamma * q_targ

            # ----- critics
            z = self.fusion_encoder(rgb, s, aug_shift=self.vcfg.aug_random_shift)
            q1_pred = self.q1(z, a)
            q2_pred = self.q2(z, a)
            q_loss = F.smooth_l1_loss(q1_pred, y) + F.smooth_l1_loss(q2_pred, y)

            self.opt_q.zero_grad(set_to_none=True)
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()) + list(self.fusion_encoder.parameters()), 10.0
            )
            self.opt_q.step()

            # ----- actor
            # z = self.fusion_encoder(rgb, s, aug_shift=self.vcfg.aug_random_shift)
            # a_new, logp_new, _ = self.actor.sample(z)
            # a_new = torch.clamp(a_new, -1.0, 1.0)
            # q1_pi = self.q1(z, a_new)
            # q2_pi = self.q2(z, a_new)
            # q_pi = torch.min(q1_pi, q2_pi)
            # pi_loss = (self.alpha.detach() * logp_new - q_pi).mean()

            # self.opt_actor.zero_grad(set_to_none=True)
            # pi_loss.backward()
            # self.opt_actor.step()

            z = self.fusion_encoder(rgb, s, aug_shift=self.vcfg.aug_random_shift).detach()
            a_new, logp_new, _ = self.actor.sample(z)
            a_new = torch.clamp(a_new, -1.0, 1.0)
            q1_pi = self.q1(z, a_new)
            q2_pi = self.q2(z, a_new)
            q_pi = torch.min(q1_pi, q2_pi)
            pi_loss = (self.alpha.detach() * logp_new - q_pi).mean()

            self.opt_actor.zero_grad(set_to_none=True)
            pi_loss.backward()
            self.opt_actor.step()

            # ----- temperature
            alpha_loss = -(self.log_alpha * (logp_new.detach() + self.target_entropy)).mean()
            self.opt_alpha.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.opt_alpha.step()

            # ----- targets polyak
            with torch.no_grad():
                for p, pt in zip(self.q1.parameters(), self.q1_targ.parameters()):
                    pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
                for p, pt in zip(self.q2.parameters(), self.q2_targ.parameters()):
                    pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

            self.q_losses.append(float(q_loss.item()))
            self.pi_losses.append(float(pi_loss.item()))

        self.last_log.update({
            "q_loss": float(self.q_losses[-1]),
            "pi_loss": float(self.pi_losses[-1]),
            "alpha": float(self.alpha.item())
        })

    # ------------- evaluate (deterministic policy)
    @torch.no_grad()
    def evaluate(self, env, episodes: int = 5, render: bool = False, max_steps: Optional[int] = None) -> float:
        if max_steps is None:
            if hasattr(env, "max_episode_length") and isinstance(env.max_episode_length, int):
                max_steps = env.max_episode_length
            else:
                max_steps = 2000

        returns = []
        for _ in range(episodes):
            obs_dict = env.reset()[0]
            state, rgb = parse_obs(obs_dict)
            ep_ret = torch.zeros(env.num_envs, device=device)
            done_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

            for t in range(max_steps):
                act, _ = self._select_action(state, rgb, deterministic=True)
                next_obs_dict, rew, dones, info, _ = env.step(act)
                ep_ret += rew * (~done_mask)
                done_mask |= dones
                state, rgb = parse_obs(next_obs_dict)
                if bool(done_mask.all()):
                    break
                if render and hasattr(env, "render"):
                    env.render()

            returns.append(ep_ret.mean().item())
        return float(sum(returns) / len(returns))

    # ------------- train loop
    def train(self, total_env_steps: int, log_interval_updates: int = 10):
        os.makedirs(self.save_dir, exist_ok=True)

        obs_dict = self.env.reset()[0]
        state, rgb = parse_obs(obs_dict)
        rollout_rew_accum = 0.0
        rollout_step_accum = 0
        t0 = time.time()
        counter = 0

        while self.ckpt.step_counter < total_env_steps:
            # act
            if self.ckpt.step_counter < self.start_random_steps:
                act = torch.empty((self.env.num_envs, self.act_dim), device=device).uniform_(-0.5, 0.5)
            else:
                act, _ = self._select_action(state, rgb, deterministic=False)

            # step env
            next_obs_dict, rew, dones, info, _ = self.env.step(act)
            next_state, next_rgb = parse_obs(next_obs_dict)

            # ----- done_bootstrap: 1 => НЕ бутстрэпим (истинный терминал)
            terminated = None
            truncated  = None
            if isinstance(info, dict):
                terminated = info.get("terminated", None)
                truncated  = info.get("time_outs", info.get("truncated", None))

            # базово берём dones как terminated, если terminated нет
            if isinstance(terminated, torch.Tensor):
                d_bootstrap = terminated.to(dtype=torch.float32)
            else:
                d_bootstrap = dones.to(dtype=torch.float32)

            # привести ВСЁ к (N,1), чтобы не было broadcasting
            d_bootstrap = d_bootstrap.view(-1, 1)

            # если есть truncated/time_outs: при усечении бутстрэпить (т.е. d=0)
            if isinstance(truncated, torch.Tensor):
                trunc = truncated.view(-1, 1).to(dtype=torch.bool)
                d_bootstrap = torch.where(trunc, torch.zeros_like(d_bootstrap), d_bootstrap)

            done_bootstrap = d_bootstrap  # (N,1)
            # log rollouts
            rollout_rew_accum += float(rew.mean().item())
            rollout_step_accum += 1

            # to uint8 NHWC for buffer
            rgb_uint8 = rgb if rgb.dtype == torch.uint8 else rgb.clamp(0, 255).to(torch.uint8)
            next_rgb_uint8 = next_rgb if next_rgb.dtype == torch.uint8 else next_rgb.clamp(0, 255).to(torch.uint8)

            # add to replay
            self.replay.add(state, rgb_uint8, act, rew, next_state, next_rgb_uint8, done_bootstrap)

            # advance
            state, rgb = next_state, next_rgb

            # updates
            self._update()

            self.ckpt.step_counter += self.env.num_envs

            # logs
            if (self.ckpt.step_counter // self.env.num_envs) % log_interval_updates == 0:
                sps = self.ckpt.step_counter / max(1.0, time.time() - t0)
                wall_m = (time.time() - t0) / 60.0
                print(
                    f"steps={self.ckpt.step_counter}  "
                    f"avgR={rollout_rew_accum / max(1, rollout_step_accum):.3f}  "
                    f"q={self.last_log.get('q_loss', float('nan')):.3f}  "
                    f"pi={self.last_log.get('pi_loss', float('nan')):.3f}  "
                    f"alpha={self.last_log.get('alpha', float('nan')):.3f}  "
                    f"SPS={sps:.0f}  t={wall_m:.1f}m"
                )
                self.rewards.append(rollout_rew_accum / max(1, rollout_step_accum))
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

                    with open(os.path.join(self.save_dir, "saves.txt"), "w") as output:
                        output.write(str(self.rewards) + "\n")
                        output.write(str(self.q_losses) + "\n")
                        output.write(str(self.pi_losses) + "\n")

    # ------------- optional: load demos if they exist (state-only fallback)
    # def load_demo_to_buffer(self, path: str):
    #     data = np.load(path)
    #     has_rgb = ("rgb" in data) and ("next_rgb" in data)
    #     obs = torch.tensor(data["obs"], dtype=torch.float32, device=device)
    #     actions = torch.tensor(data["actions"], dtype=torch.float32, device=device)
    #     rewards = torch.tensor(data["rewards"], dtype=torch.float32, device=device).squeeze(-1)
    #     next_obs = torch.tensor(data["next_obs"], dtype=torch.float32, device=device)
    #     dones = torch.tensor(data["dones"], dtype=torch.float32, device=device).squeeze(-1)

    #     if has_rgb:
    #         rgb = torch.tensor(data["rgb"], dtype=torch.uint8, device=device)
    #         next_rgb = torch.tensor(data["next_rgb"], dtype=torch.uint8, device=device)
    #     else:
    #         # if demos are state-only, synthesize blank images (not ideal, but keeps API)
    #         N = obs.shape[0]
    #         H, W = self.vcfg.img_size
    #         rgb = torch.zeros(N, H, W, 3, dtype=torch.uint8, device=device)
    #         next_rgb = torch.zeros_like(rgb)

    #     # chunked add to avoid OOM
    #     bs = 4096
    #     for i in range(0, obs.shape[0], bs):
    #         j = min(i + bs, obs.shape[0])
    #         self.replay.add(obs[i:j], rgb[i:j], actions[i:j], rewards[i:j], next_obs[i:j], next_rgb[i:j], dones[i:j])

    #     print(f"[Buffer] Loaded {obs.shape[0]} transitions from demo into replay buffer.")

    def load_demo_to_buffer(self, path: str):
        data = np.load(path, mmap_mode='r')   # лениво
        has_rgb = ("rgb" in data) and ("next_rgb" in data)

        obs      = data["obs"]        # shape [N, S] или что у тебя в демо
        actions  = data["actions"]
        rewards  = data["rewards"]
        next_obs = data["next_obs"]
        dones    = data["dones"]
        N = obs.shape[0]

        H, W = self.vcfg.img_size

        bs = 4096   # можно 2048, если RAM ограничена
        for i in range(0, N, bs):
            j = min(i + bs, N)

            s   = torch.from_numpy(obs[i:j]).to(torch.float32)         # CPU
            a   = torch.from_numpy(actions[i:j]).to(torch.float32)     # CPU
            r   = torch.from_numpy(rewards[i:j]).to(torch.float32).squeeze(-1)  # CPU
            ns  = torch.from_numpy(next_obs[i:j]).to(torch.float32)    # CPU
            dn  = torch.from_numpy(dones[i:j]).to(torch.float32).squeeze(-1)    # CPU

            if has_rgb:
                rgb_chunk   = torch.from_numpy(data["rgb"][i:j]).to(torch.uint8)        # CPU
                n_rgb_chunk = torch.from_numpy(data["next_rgb"][i:j]).to(torch.uint8)   # CPU
            else:
                # создаём только МАЛЕНЬКИЕ чанки нулей, а не весь N
                rgb_chunk   = torch.zeros(j - i, H, W, 3, dtype=torch.uint8)   # CPU
                n_rgb_chunk = torch.zeros_like(rgb_chunk)

            # add() у нас CPU-friendly и сам ресайзит до (H,W)
            self.replay.add(s, rgb_chunk, a, r, ns, n_rgb_chunk, dn)

        print(f"[Buffer] Loaded {N} transitions from '{path}' into replay (CPU).")
