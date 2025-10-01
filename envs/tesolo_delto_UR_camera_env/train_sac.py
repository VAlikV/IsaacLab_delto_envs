from isaaclab.app import AppLauncher

# Запускаем Kit headless, без камер и стрима
app_launcher = AppLauncher(
    headless=True,
    enable_cameras=True,
    enable_livestream=False,
)
simulation_app = app_launcher.app


from agents.sac2_vision_fusion import SAC, parse_obs, device, VisionCfg
from delto_env import DeltoEnvCfg, DeltoEnv

if __name__ == "__main__":

    cfg = DeltoEnvCfg()
    env = DeltoEnv(cfg)

    obs0 = env.reset()[0]
    obs_dim = parse_obs(obs0)[0].shape[-1]
    act_dim = cfg.action_space

    v_cfg = VisionCfg(
        img_size=(84, 84),      # (H, W)
        aug_random_shift=0,       # пиксельный сдвиг ↑ пропорционально размеру
        encoder_out_dim=32,
    )

    agent = SAC(
        env,
        obs_dim, act_dim,
        vision_cfg=v_cfg,
        gamma=0.99,
        tau=0.003,
        lr_actor=3e-4,
        lr_critic=5e-5,             
        lr_alpha=3e-4,
        target_entropy= -act_dim,
        buffer_capacity=75_000,
        batch_size=1024,
        updates_per_step=1,         
        start_random_steps=0, 
        save_dir="runs/delto_UR_hand_smallN",
    )

    # agent.load_demo_to_buffer("trajectories/4th_try.npz")

    agent.train(total_env_steps=2_000_000, log_interval_updates=10)
