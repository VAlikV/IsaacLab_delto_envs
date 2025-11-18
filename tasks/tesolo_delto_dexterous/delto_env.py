
from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
# from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import ContactSensor

# from isaaclab.utils.math import quat_to_rot_mats
from isaaclab.utils.math import quat_apply

from .utils.utils import sample_object_point_cloud, object_point_cloud_b

from isaaclab.markers import VisualizationMarkers

##
# Configuration
##

from .delto_env_cfg import DeltoEnvCfg

   
# =====================================================================================================
# =====================================================================================================
# =====================================================================================================

class DeltoEnv(DirectRLEnv):
    cfg: DeltoEnvCfg

    def __init__(self, cfg: DeltoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        N = self.scene.num_envs

        self.action_scale = torch.tensor(self.cfg.action_scale, device=self.device).unsqueeze(0)

        # Координаты центров сред
        self.env_origins = self.scene.env_origins

        # -------------------------------------------

        self.hand = self.scene.articulations["hand"]

        self.hand_joint_ids, _ = self.hand.find_joints(cfg.hand_joint_names)
        self.arm_joint_ids, _ = self.hand.find_joints(cfg.arm_joint_names)
        self.ft_ids = [self.hand.body_names.index(n) for n in self.cfg.ft_names]
        
        self.object = self.scene.rigid_objects["object"]
        self.table = self.scene.rigid_objects["table"]

        # self.cam = self.scene.sensors["camera"]

        # -------------------------------------------

        self.joint_pos = self.hand.data.joint_pos
        self.joint_vel = self.hand.data.joint_vel   

        # -------------------------------------------

        self.contact_treshold = 1.5

        # -------------------------------------------

        self.arm_start_position = torch.tensor(self.cfg.arm_position, device=self.device)*np.pi/180
        self.hand_start_position = torch.tensor(self.cfg.hand_position, device=self.device)*np.pi/180

        self.robot_start_position = torch.cat([self.arm_start_position, self.hand_start_position])

        # -------------------------------------------

        self.arm_upper_limits = torch.tensor(self.cfg.arm_upper_limits, device=self.device)*np.pi/180
        self.arm_lower_limits = torch.tensor(self.cfg.arm_lower_limits, device=self.device)*np.pi/180

        self.hand_upper_limits = torch.tensor(self.cfg.hand_upper_limits, device=self.device)*np.pi/180
        self.hand_lower_limits = torch.tensor(self.cfg.hand_lower_limits, device=self.device)*np.pi/180

        self.robot_upper_limits = torch.cat([self.arm_upper_limits, self.hand_upper_limits])
        self.robot_lower_limits = torch.cat([self.arm_lower_limits, self.hand_lower_limits])

        # -------------------------------------------

        self.raw_actions = torch.zeros(N, self.cfg.action_space, device=self.device)
        self.raw_prev_actions = torch.zeros(N, self.cfg.action_space, device=self.device)
         
        self.target_pos = torch.zeros(N, self.cfg.action_space, device=self.device)
        self.target_pos[:] = self.robot_start_position

        self.target_height = 0.4

        # -------------------------------------------

        self.per_env_timeout = torch.full((self.scene.num_envs,), self.max_episode_length, device=self.device)

        # -------------------------------------------

        self.history_length = 5
        self.flatten_history_dim = True     # как в ObsGroup.flatten_history_dim
        self._hist_inited = False           # ленивое создание буфера истории
        self._obs_hist_policy = None        # [N, H, obs_dim]

        self._obs_hist_policy = torch.zeros((N, self.history_length, self.cfg.observation_space // 5), device=self.device)


# =====================================================================================================

    def _setup_scene(self):

        spawn_ground_plane("/World/ground", GroundPlaneCfg())

        self.hand = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["hand"] = self.hand

        self.object = RigidObject(self.cfg.object_cfg)
        self.scene.rigid_objects["object"] = self.object

        self.table = RigidObject(self.cfg.table_cfg)
        self.scene.rigid_objects["table"] = self.table

        self.visualizer = VisualizationMarkers(self.cfg.ray_cfg)

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        for name in self.cfg.ft_names:
            self.scene.sensors[name] = ContactSensor(self.cfg.contact_sensors[name])    # доступ: self.scene.sensors["robot0_ffdistal"]

        self.pcd = sample_object_point_cloud(num_envs=self.scene.num_envs, num_points=25, prim_path="/World/envs/env_.*/Object", device=self.device)

# =====================================================================================================

    def _pre_physics_step(self, actions: torch.Tensor) -> None:

        self.prew_raw_actions = self.raw_actions.clone()
        self.raw_actions = actions.clone()

        self.actions = self.action_scale * actions.clone()

        # print(self.actions)
        
        self.target_pos = self.target_pos + self.actions

        self.target_pos = torch.clip(self.target_pos, self.robot_lower_limits, self.robot_upper_limits)

        # print(self.target_pos)

# =====================================================================================================

    def _apply_action(self):

        self.hand.set_joint_position_target(self.target_pos)
        # self.hand.set_joint_position_target(self.target_pos, joint_ids=self.hand_joint_ids)

# =====================================================================================================

    def _get_observations(self):

        fforce, flags, fpose = self._read_contacts(threshold=self.contact_treshold)  # [N, c1], [N, c2], [N, c3]
        points = object_point_cloud_b(self.object, self.pcd) - self.env_origins.unsqueeze(1)

        parts = [
            self.joint_pos.float(),                 # [N, J]
            fforce.float(),                         # [N, F]
            fpose.float().flatten(start_dim=1),     # [N, P]
            points.float().flatten(start_dim=1),    # [N, K]
        ]
        obs_policy = torch.cat(parts, dim=1)        # [N, obs_dim]
        N, obs_dim = obs_policy.shape

        # сдвиг окна истории
        self._obs_hist_policy = torch.roll(self._obs_hist_policy, shifts=-1, dims=1)
        self._obs_hist_policy[:, -1, :] = obs_policy

        # print(points.float().mean(dim=1))
        # print()
        # print(object_com)
        # print()

        # подготовка выхода по аналогии с flatten_history_dim / concatenate_terms
        if self.flatten_history_dim:
            # [N, H, obs_dim] → [N, H*obs_dim]
            obs_policy_out = self._obs_hist_policy.reshape(N, self.history_length * obs_dim)
        else:
            # вернём с явной размерностью истории (напр. для RNN)
            obs_policy_out = self._obs_hist_policy  # [N, H, obs_dim]

        return {
            "policy": obs_policy_out,
        }
        
# =====================================================================================================

    def _get_rewards(self) -> torch.Tensor:

        fforce, flags, fpose = self._read_contacts(threshold=self.contact_treshold)  # силы и флаги в world frame
   
        object_com_w = self.object.data.body_com_pos_w[:, 0, 0:3]
        object_com   = object_com_w - self.env_origins

        # 1) Расстояние
        object_distance = (fpose - object_com.unsqueeze(1)).norm(dim=2).max(dim=1).values
        r_approach = 1.0 * (1 - torch.tanh(object_distance / 0.4))
        # r_approach = -1.0 * object_distance

        # 2) Контакты
        active_cnt = (flags > 0.5).float().sum(dim=1)
        r_contact_hold = 0.3 * active_cnt

        # print(active_cnt)

        # 3) Награда за высоту
        obj_r_h = self.object.data.default_root_state[:,2]
        # obj_h = torch.abs(object_com[:, 2] - obj_r_h)
        obj_dh = object_com[:, 2] - obj_r_h
        obj_h = torch.where(obj_dh > 0.0, obj_dh, 0.0)
        r_h = 1.0 * torch.tanh(obj_h / self.cfg.success_height) * (active_cnt == 3).float()

        # 4) Штраф за силу сжатия
        max_fforce, _ = torch.max(fforce, dim=1)
        max_fforce_treshed = torch.where(max_fforce > 10.0, -max_fforce, max_fforce)
        force_penalty = 0.3 * max_fforce_treshed

        # 5) Штраф за скорость 
        vel = torch.sum(torch.square(self.raw_prev_actions - self.raw_actions), dim=1).clamp(-1000, 1000)
        # vel = torch.norm(self.prev_actions - self.actions, dim=1).clamp(-1000, 1000)
        r_vel = -0.001 * vel

        # 6) Штраф за действие
        # effort_norm = self.actions.norm(dim=1).clamp(-1000, 1000)
        effort_norm = torch.sum(torch.square(self.raw_actions), dim=1).clamp(-1000, 1000)
        r_effort = -0.001 * effort_norm

        # 7) Штраф за движение пальцев
        # hand_action_magnitude = self.actions[:, self.hand_joint_ids].norm(dim=1)
        # r_finger_smoothness = -0.01 * hand_action_magnitude

        rew = r_approach + r_contact_hold + force_penalty + r_vel + r_effort + r_h

        success_mask = (object_com[:, 2] >= self.cfg.success_height) & (active_cnt == 3)

        termination_mask = self._check_termination()

        rew = rew + 30.0 * success_mask.float() + 0.0 * termination_mask.float()

        # print("rew ", rew)
        # print("r_approach ", r_approach)
        # print("r_contact_hold ", r_contact_hold)
        # print("force_penalty ", force_penalty)
        # print("r_vel ", r_vel)
        # print("r_effort ", r_effort)
        # print("r_h ", r_h)
        # print()
    
        return rew
    
# =====================================================================================================

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    
        terminations = self._check_termination()

        truncations = self.episode_length_buf >= self.per_env_timeout

        return terminations, truncations
    
# =====================================================================================================

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = torch.zeros((len(env_ids), 26), device=self.device)
        joint_vel = torch.zeros((len(env_ids), 26), device=self.device)
        
        noise = torch.randint(-3, 4, (len(env_ids), 26), device=self.device)*np.pi/180
        joint_pos[:] = self.robot_start_position + noise
        self.target_pos[env_ids] = self.robot_start_position + noise

        joint_pos = torch.clip(joint_pos, self.robot_lower_limits, self.robot_upper_limits)
        self.target_pos[env_ids] = torch.clip(self.target_pos[env_ids], self.robot_lower_limits, self.robot_upper_limits)

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.raw_prev_actions[env_ids] = torch.zeros((len(env_ids), 26), device=self.device)

        low = int(0.7 * self.max_episode_length)
        high = int(self.max_episode_length)
        self.per_env_timeout[env_ids] = torch.randint(low, high+1, (len(env_ids),), device=self.device)
        
        self.hand.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.hand.set_joint_position_target(self.target_pos[env_ids], env_ids=env_ids)

        self._obs_hist_policy[env_ids].zero_()

        self._reset_object_on_table(env_ids)

        # self.cam.reset(env_ids)

    # =====================================================================================================

    def _reset_object_on_table(self, env_ids):
        # Получаем дефолтное состояние объекта для заданных сред
        root_state = self.object.data.default_root_state[env_ids].clone()

        # Шум по X и Y
        noise_xy = 0.3 * (torch.rand((len(env_ids), 2), device=self.device) - 0.5)
        noise_z = 0.2 * torch.rand((len(env_ids), 1), device=self.device)
        root_state[:, 0] += noise_xy[:, 0]  # x
        root_state[:, 1] += noise_xy[:, 1]  # y
        root_state[:, 2] += noise_z[:, 0]  # z

        # === Случайная ориентация вокруг оси Z ===
        yaw_angles = 2 * np.pi * torch.rand((len(env_ids),), device=self.device)
        half_yaw = 0.5 * yaw_angles

        quat_z = torch.zeros((len(env_ids), 4), device=self.device)
        quat_z[:, 0] = torch.cos(half_yaw)  # w
        quat_z[:, 3] = torch.sin(half_yaw)  # z

        root_state[:, 3:7] = quat_z  # quaternion

        # Обнуляем линейные и угловые скорости
        root_state[:, 7:13] = 0.0

        # Сдвигаем объект в центр своей среды
        root_state[:, 0:3] += self.scene.env_origins[env_ids]

        # Записываем обратно в симулятор
        self.object.write_root_state_to_sim(root_state, env_ids)

# ===========================================================================
# ===========================================================================
# ===========================================================================

    def _read_contact_forces(self) -> torch.Tensor:
        forces_per_finger = []

        for name in self.cfg.ft_names:
            f = self.scene.sensors[name].data.force_matrix_w.sum(dim=1)  # shape: (N, 3) или (N, H, 3) при history_length>1
            if f.ndim == 3:
                f = f[:, -1, :]  # берём последний сэмпл истории

            fnorm = torch.linalg.norm(f, dim=-1)
            forces_per_finger.append(fnorm)

        return torch.stack(forces_per_finger, dim=1)  # (N, 5)
    
    # =====================================================================================================        

    def _read_contacts(self, threshold: float = 1.0):
        F = self._read_contact_forces()
        flags = (F > threshold).float()

        ft_pos_w = self.hand.data.body_pos_w[:, self.ft_ids, :]         # (N,5,3) world
        ft_pos   = ft_pos_w - self.env_origins.unsqueeze(1)

        return F, flags, ft_pos
    
# ===========================================================================
# ===========================================================================
# ===========================================================================      

    def _check_termination(self) -> torch.Tensor:
        object_pos = self.object.data.body_com_pos_w[:, 0, 0:3] - self.env_origins  # [num_envs, 3]
        
        # Примеры допустимых границ
        x_min, x_max = -2.5, 2.5
        y_min, y_max = -3.1, 0.0
        z_min, z_max = 0.0, 2.0

        out_of_bounds = (
            (object_pos[:, 0] < x_min) | (object_pos[:, 0] > x_max) |
            (object_pos[:, 1] < y_min) | (object_pos[:, 1] > y_max) |
            (object_pos[:, 2] < z_min) | (object_pos[:, 2] > z_max)
        )

        return out_of_bounds
