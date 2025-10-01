from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False, enable_cameras=True, enable_livestream=False)
simulation_app = app_launcher.app

import torch
from agents.sac2_vision_fusion import (
    TanhGaussianPolicy, FusionEncoder, VisionCfg, preprocess_rgb, parse_obs, device
)
from delto_env import DeltoEnvCfg, DeltoEnv

CKPT_PATH = "runs/delto_UR_hand_smallN/last.pt"   # или last.pt

@torch.no_grad()
def run_eval(num_episodes=5, render=False, max_steps=None):
    # 1) Env
    cfg = DeltoEnvCfg()
    env = DeltoEnv(cfg, render_mode=None)

    # 2) Получаем размерности
    obs0 = env.reset()[0]
    state0, rgb0 = parse_obs(obs0)                   
    state_dim = state0.shape[-1]
    try:
        act_dim = env.action_space.shape[1]
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@2", env.action_space.shape)
    except Exception:
        act_dim = len(getattr(cfg, "actuated_joint_names", [])) or int(getattr(cfg, "action_space"))

    # 3) Vision config — ДОЛЖЕН совпадать с тренировочным!
    vcfg = VisionCfg(
        img_size=(84, 84),          
        frame_stack=1,
        grayscale=False,
        aug_random_shift=0,
        encoder_out_dim=32,
    )

    # 4) Сборка encoder + actor
    fusion_encoder = FusionEncoder(state_dim, vcfg).to(device).eval()
    fused_dim = fusion_encoder.fused_dim
    actor = TanhGaussianPolicy(fused_dim, act_dim).to(device).eval()

    # 5) Загрузка чекпоинта (после патча A: obj_0=encoder, obj_1=actor)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    if "obj_0" in ckpt and "obj_1" in ckpt:
        fusion_encoder.load_state_dict(ckpt["obj_0"], strict=False)
        actor.load_state_dict(ckpt["obj_1"], strict=False)
    else:
        # Старые чекпоинты (где encoder не сохранялся): грузим только actor
        actor.load_state_dict(ckpt.get("obj_0", ckpt))
        print("[WARN] Checkpoint has no fusion_encoder weights; using randomly initialized encoder.")

    # 6) max_steps
    if max_steps is None:
        max_steps = env.max_episode_length if hasattr(env, "max_episode_length") else 2000

    # 7) Эвал
    returns = []
    for _ in range(num_episodes):
        obs_dict = env.reset()[0]
        state, rgb_uint8 = parse_obs(obs_dict)
        ep_ret = torch.zeros(env.num_envs, device=device)
        done_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

        t = 0
        while True:
            rgb_nchw = preprocess_rgb(rgb_uint8.to(device), vcfg)
            z = fusion_encoder(rgb_nchw, state.to(device), aug_shift=0)
            act = actor.act_mean(z)
            act = torch.clamp(act, -1.0, 1.0)

            print(act[0])

            next_obs_dict, rew, dones, info, _ = env.step(act)
            ep_ret += rew.to(device) * (~done_mask)
            done_mask |= dones.to(device)

            state, rgb_uint8 = parse_obs(next_obs_dict)

            if bool(done_mask.all()) or (t >= max_steps - 1):
                break
            if render and hasattr(env, "render"):
                env.render()
            t += 1

        returns.append(ep_ret.mean().item())

    print(f"[SAC] Eval avg return over {num_episodes} episodes: {sum(returns)/len(returns):.3f}")

if __name__ == "__main__":
    run_eval(num_episodes=5, render=False)
