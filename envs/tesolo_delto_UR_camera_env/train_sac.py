from isaaclab.app import AppLauncher

# Запускаем Kit headless, без камер и стрима
app_launcher = AppLauncher(
    headless=True,
    enable_cameras=False,
    enable_livestream=False,
)
simulation_app = app_launcher.app


from sac2 import SAC, obs_to_tensor, device
from delto_env import DeltoEnvCfg, DeltoEnv

if __name__ == "__main__":
    # Ожидаем, что hand_env.py лежит в PYTHONPATH или рядом, и содержит HandEnv/HandEnvCfg. :contentReference[oaicite:3]{index=3}

    cfg = DeltoEnvCfg()
    env = DeltoEnv(cfg)  # в IsaacLab DirectRLEnv совместимо с твоим PPO

    # определяем размерности автоматически
    obs0 = env.reset()[0]
    obs_dim = obs_to_tensor(obs0).shape[-1]
    act_dim = cfg.action_space  # у тебя action_space = 20 в конфиге среды. :contentReference[oaicite:4]{index=4}

    # agent = SAC(env, obs_dim, act_dim,
    #             gamma=0.98, tau=0.01,
    #             lr_actor=3e-4, lr_critic=1e-4, lr_alpha=3e-4,
    #             batch_size=256, updates_per_step=1, start_random_steps=2048,
    #             save_dir="runs/hand_grasp_then_lift")
    
    agent = SAC(env, obs_dim, act_dim,
        gamma=0.99, tau=0.02,
        lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
        target_entropy= -0.3 * act_dim,
        buffer_capacity=3_000_000,
        batch_size=1536,               # 1024–2048 ок, я бы начал с 1536
        updates_per_step=2,            # можно 3, если буфер быстро растёт
        start_random_steps=2048*64,    # потом вернёшь к 2048*32
        save_dir="runs/delto_UR_hand")
    


    agent.load_demo_to_buffer("trajectories/1st_try.npz")

    agent.train(total_env_steps=20_000_000, log_interval_updates=10)
