from isaaclab.app import AppLauncher

app_launcher = AppLauncher(
    headless=False,
    enable_cameras=False,
    enable_livestream=False,
)
simulation_app = app_launcher.app

import torch
from sac2 import TanhGaussianPolicy, obs_to_tensor, device
from delto_env import DeltoEnvCfg, DeltoEnv

CKPT_PATH = "runs/delto_UR_hand/best.pt"   # или last.pt

@torch.no_grad()
def run_eval(num_episodes=5, render=False, max_steps=None):
    cfg = DeltoEnvCfg()
    env = DeltoEnv(cfg, render_mode=None)

    obs0 = env.reset()[0]
    obs_dim = obs_to_tensor(obs0).shape[-1]
    act_dim = len(cfg.actuated_joint_names) if hasattr(cfg, "actuated_joint_names") else getattr(cfg, "action_space")

    actor = TanhGaussianPolicy(obs_dim, act_dim).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    if max_steps is None:
        max_steps = env.max_episode_length if hasattr(env, "max_episode_length") else 2000

    returns = []
    for _ in range(num_episodes):
        obs_dict = env.reset()[0]
        obs = obs_to_tensor(obs_dict)
        ep_ret = torch.zeros(env.num_envs, device=device)
        done_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

        t = 0
        while True:
            act = actor.act_mean(obs)
            act = torch.clamp(act, -1.0, 1.0)
            next_obs, rew, dones, info, _ = env.step(act)
            ep_ret += rew * (~done_mask)
            done_mask |= dones
            obs = obs_to_tensor(next_obs)

            if done_mask.all() or (t >= max_steps):
                break
            if render and hasattr(env, "render"):
                env.render()
            t += 1

        returns.append(ep_ret.mean().item())

    print(f"[SAC] Eval avg return over {num_episodes} episodes: {sum(returns)/len(returns):.3f}")

if __name__ == "__main__":
    run_eval(num_episodes=500, render=False)