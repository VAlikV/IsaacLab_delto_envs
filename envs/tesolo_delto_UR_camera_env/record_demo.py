import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from delto_env import DeltoEnv, DeltoEnvCfg 

import numpy as np
import torch
from sac2 import obs_to_tensor  # если SAC в sac2.py
import time
import os
import socket

# ============================================================================================================
# ============================================================================================================
# ============================================================================================================

def record_manual_trajectories(env, action, max_steps=300, save_path="trajectories/manual_demo.npz"):
    traj_data = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "next_obs": [],
        "dones": []
    }

    obs_dict = env.reset()[0]
    obs = obs_to_tensor(obs_dict)
    
    for s in range(max_steps):

        print(s)
        
        # ===== Получить действие вручную (например, от IK/геймпада)
        action = actor_fn(action)  # <-- твоя реализация телеуправления
        # action = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(obs.device)

        # ===== Выполнить шаг
        next_obs_dict, reward, done, info, _ = env.step(action)
        next_obs = obs_to_tensor(next_obs_dict)

        # ===== Сохраняем всё
        traj_data["obs"].append(obs.cpu().numpy())
        traj_data["actions"].append(action.cpu().numpy())
        traj_data["rewards"].append(reward.cpu().numpy())
        traj_data["next_obs"].append(next_obs.cpu().numpy())
        traj_data["dones"].append(done.cpu().numpy())

        obs = next_obs
        obs_dict = next_obs_dict

    for k in traj_data:
        traj_data[k] = np.concatenate(traj_data[k], axis=0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, **traj_data)
    print(f"[Saved] Manual trajectories saved to {save_path}")

# ============================================================================================================

def actor_fn(action):
    try:
        data, address = server_socket.recvfrom(1024)

        data = list(map(float, (str(data)[3:-2].split(", "))))

        # print(data)
        action[:,0:20] = torch.tensor(data)

        return action
        
    except BlockingIOError:

        return action
    
# ============================================================================================================
# ============================================================================================================
# ============================================================================================================

if __name__ == "__main__":
    
    server_ip = '127.0.0.1'  
    server_port = 8081  

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  
    recv_buf_size = 1024
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_buf_size)
    server_socket.bind((server_ip, server_port)) 
    server_socket.settimeout(0.001)
    server_socket.setblocking(False)

    # Создаем среду
    env = DeltoEnv(DeltoEnvCfg())

    # Сброс среды
    obs = env.reset()
    print("Initial observation:", obs)

    start = torch.zeros([env.cfg.num_env, env.cfg.action_space])

    obs, rewards, dones, info, _ = env.step(start)

    record_manual_trajectories(env, start, max_steps=1_500, save_path="trajectories/manual_demo.npz")



