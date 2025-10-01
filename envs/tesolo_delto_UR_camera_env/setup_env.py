import argparse

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(
    headless=False,
    enable_cameras=True,
    enable_livestream=False,
)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from delto_env import DeltoEnv, DeltoEnvCfg 
import time
import os
import numpy as np
# import cv2
from PIL import Image
import matplotlib.pyplot as plt

##
# Pre-defined configs
##

import socket  


if __name__ == "__main__":
    
    # Для управления по UDP
    # server_ip = '192.168.68.122' 
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
    
    pos = torch.zeros([env.cfg.num_env, env.cfg.action_space])

    # Для отрисовки кадров
    plt.ion()
    fig, ax = plt.subplots()
    im = None
    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(fig.bbox)
    i = 0

    while(True):
        
        try:
            data, address = server_socket.recvfrom(1024)

            data = list(map(float, (str(data)[3:-2].split(", "))))

            # print(data)
            pos[:,0:20] = torch.tensor(data)
            
        except BlockingIOError:
            # данных пока нет
            pass

        obs, rewards, dones, info, _ = env.step(pos)

        # print("Joints pos: ", obs["state"]["joints_pos"])
        # print("Obj pos: ", obs["state"]["object_pos"])
        # print("Forces: ", obs["state"]["contact_forces"])
        # print("Flags: ", obs["state"]["contact_flags"])
        # print("Tips pos: ", obs["state"]["finger_tips_pos"])
        # print("Hand open: ", obs["state"]["hand_open"])
        # print("Reward", rewards)
        # print(obs["rgb"])

        arr = obs["rgb"][0].cpu().numpy()

        fig.canvas.restore_region(bg)
        if im is None:
            im = ax.imshow(arr, animated=True)
            ax.axis('off')
        else:
            im.set_data(arr)
            ax.draw_artist(im)

        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()


        print(i)
        i += 1
        print("====================================")

        time.sleep(0.05)