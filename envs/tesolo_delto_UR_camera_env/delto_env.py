
from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
# from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

from isaaclab.sensors import ContactSensorCfg, ContactSensor
from isaaclab.sensors import Camera, CameraCfg

# from isaaclab.utils.math import quat_to_rot_mats
from isaaclab.utils.math import quat_apply

##
# Configuration
##

from ur_delto_cfg import DELTO_CFG


# =====================================================================================================
# =====================================================================================================
# =====================================================================================================

@configclass
class DeltoEnvCfg(DirectRLEnvCfg):

    # ======================================================================= env params
    decimation = 2
    episode_length_s = 5.0
    action_space = 20
    observation_space = 157  # (full)
    state_space = 0
    action_scale = 1
    asymmetric_obs = False
    obs_type = "full"
    num_env = 16

    # ======================================================================= simulation

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )

    # ======================================================================= robot

    robot_cfg: ArticulationCfg = DELTO_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace()
    # robot_cfg: ArticulationCfg = DELTO_CFG

    arm_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint"
    ]

    hand_joint_names = [
        "rj_dg_1_1",
        "rj_dg_1_2",
        "rj_dg_1_3",
        "rj_dg_1_4",
        "rj_dg_2_1",
        "rj_dg_2_2",
        "rj_dg_2_3",
        "rj_dg_2_4",
        "rj_dg_3_1",
        "rj_dg_3_2",
        "rj_dg_3_3",
        "rj_dg_3_4",
        "rj_dg_4_1",
        "rj_dg_4_2",
        "rj_dg_4_3",
        "rj_dg_4_4",
        "rj_dg_5_1",
        "rj_dg_5_2",
        "rj_dg_5_3",
        "rj_dg_5_4"
    ]

    hand_position = [30, 0, 0, 0, 0, -50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    start_position = [-90.0, -75.0, 120.0, -50.0, 90.0, -90.0]
    lift_delta_deg = [-0.0, -10.0, -10.0, 20.0, 0.0, 0.0]

    # ======================================================================= sensors

    ft_names = [
        "rl_dg_1_4",
        "rl_dg_2_4",
        "rl_dg_3_4",
        "rl_dg_4_4",
        "rl_dg_5_4",
    ]

    # fingers_names = [
    #     "rl_dg_1_4/rl_dg_1_tip",
    #     "rl_dg_2_4/rl_dg_2_tip",
    #     "rl_dg_3_4/rl_dg_3_tip",
    #     "rl_dg_4_4/rl_dg_4_tip",
    #     "rl_dg_5_4/rl_dg_5_tip",
    # ]

    contact_sensors = {}
    for name in ft_names:
        contact_sensors[name] = ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/dg5f_my/{name}",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            # filter_prim_paths_expr=["/World/envs/env_.*/Cube"],
            track_air_time=False,
        )

    # ======================================================================= objects

    # in-manipulator object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",  # один куб на среду
        spawn=sim_utils.CylinderCfg(  # используем Box вместо Cone
            radius=0.03,  # кубик 10 см
            height=0.15,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=2.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # красный
                metallic=0.1,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.23, -0.83, 0.1),
            # pos=(0.22, -1.0, 0.1),
        ),
    )
    
    table_cfg: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Table",  # один куб на среду
            spawn=sim_utils.CuboidCfg(  # используем Box вместо Cone
                size=(0.5, 0.5, 0.1),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    kinematic_enabled=True
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=15.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.1, 0.1),
                    metallic=0.1,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.30, -0.85, 0.05),
            ),
        )
    
    wall_cfg: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Wall",  # один куб на среду
            spawn=sim_utils.CuboidCfg(  # используем Box вместо Cone
                size=(0.05, 2.5, 1.5),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    kinematic_enabled=False
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.01, 0.01, 0.01),
                    metallic=0.1,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-0.7, -0.85, 1.5/2),
            ),
        )
    
    # cam_cfg = CameraCfg(
    #     prim_path="/World/envs/env_.*/Robot/ee_link/Camera",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         # focal_length=3.55, focus_distance=270.0, horizontal_aperture=5.76, vertical_aperture=3.24, clipping_range=(0.01, 20.0)
    #         focal_length=1.51, focus_distance=39.0, horizontal_aperture=5.76, vertical_aperture=4.608, clipping_range=(0.01, 20.0)

    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.05, -0.08, 0.0), rot=(0.70290398, 0.70290398, 0.07698051, 0.07698051), convention="world"),
    # )

    cam_cfg = CameraCfg(
        prim_path="/World/envs/env_.*/Table/Camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.55, focus_distance=270.0, horizontal_aperture=5.76, vertical_aperture=3.24, clipping_range=(0.01, 20.0)
            # focal_length=1.51, focus_distance=39.0, horizontal_aperture=5.76, vertical_aperture=4.608, clipping_range=(0.01, 20.0)

        ),
        offset=CameraCfg.OffsetCfg(pos=(0.3, 0.0, 0.2), rot=(0, 0, 0, 1), convention="world"),
    )

    # ======================================================================= scene

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=num_env, 
        env_spacing= 3.0, 
        replicate_physics=True
        )
    
    # === двухфазная логика
    grasp_contact_min = 3            # минимум активных пальцев
    grasp_fc_angle_deg = 110.0       # «распора» (force closure) — угол
    grasp_hold_steps = 20            # сколько тактов подряд держать хват
    grasp_max_dist = 0.07            # м: COM объекта близко к центру хвата

    lift_steps = 120                 # длительность подъёма (шагов симуляции)
    success_height = 0.12            # м: на столько поднять выше поверхности стола
    slip_grace = 10                  # допускаем кратковременную потерю контакта, шагов

    tilt_fail_deg = 55.0 
    
    #  ======================================================================= rewards

   
# =====================================================================================================
# =====================================================================================================
# =====================================================================================================

class DeltoEnv(DirectRLEnv):
    cfg: DeltoEnvCfg

    def __init__(self, cfg: DeltoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        N = self.scene.num_envs

        self.action_scale = self.cfg.action_scale

        self.env_origins = self.scene.env_origins

        # -------------------------------------------

        self.hand = self.scene.articulations["hand"]

        self.hand_joint_ids, _ = self.hand.find_joints(cfg.hand_joint_names)
        self.arm_joint_ids, _ = self.hand.find_joints(cfg.arm_joint_names)
        self.ft_ids = [self.hand.body_names.index(n) for n in self.cfg.ft_names]
        
        self.object = self.scene.rigid_objects["object"]
        self.table = self.scene.rigid_objects["table"]
        self.wall = self.scene.rigid_objects["wall"]

        # -------------------------------------------

        self.cam = Camera(cfg.cam_cfg)
        # self.cam.initialize(self.scene)

        # -------------------------------------------

        self.joint_pos = self.hand.data.joint_pos
        self.joint_vel = self.hand.data.joint_vel   

        # -------------------------------------------

        self.contact_treshold = 1.5

        self.hold_count = torch.zeros(N, dtype=torch.long, device=self.device)

        self.prev_contact_flags = torch.zeros(N, 5, device=self.device)
        self.prev_hand_open = torch.zeros(N, device=self.device)

        self.phase = torch.zeros(N, dtype=torch.long, device=self.device)
        self.phase_step = torch.zeros(N, dtype=torch.long, device=self.device)

        self.lift_hold_ok = torch.zeros(N, dtype=torch.long, device=self.device)
        self.grasp_ok_counter = torch.zeros(N, dtype=torch.long, device=self.device)
        self.slip_counter = torch.zeros(N, dtype=torch.long, device=self.device)

        # -------------------------------------------

        # хранение "замороженных" целевых поз пальцев на фазу подъёма
        self.saved_grasp_targets = torch.zeros(N, len(self.hand_joint_ids), device=self.device)

        # линейная траектория для сустава руки во время подъема
        self.arm_start = torch.zeros(N, len(self.arm_joint_ids), device=self.device)
        self.arm_goal  = torch.zeros_like(self.arm_start)

        with torch.no_grad():
            self.arm_start= self.joint_pos[:, self.arm_joint_ids]
            delta = torch.tensor(self.cfg.lift_delta_deg, device=self.device) * torch.pi/180.0
            self.arm_goal = self.arm_start + delta

        self.scripted_lift_enabled = True

# =====================================================================================================

    def _setup_scene(self):
        # prim_utils.create_prim("/World/envs/env_0/Robot", "Xform")
        spawn_ground_plane("/World/ground", GroundPlaneCfg())

        self.hand = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["hand"] = self.hand

        self.object = RigidObject(self.cfg.object_cfg)
        self.scene.rigid_objects["object"] = self.object

        self.table = RigidObject(self.cfg.table_cfg)
        self.scene.rigid_objects["table"] = self.table

        self.wall = RigidObject(self.cfg.wall_cfg)
        self.scene.rigid_objects["wall"] = self.wall

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # self.force_thresh = 1.0  # Н
        # self.ema = 0.9               # сглаживание флагов/сил, чтобы не мигало

        # print(self.cfg.contact_sensors)

        for name in self.cfg.ft_names:
            self.scene.sensors[name] = ContactSensor(self.cfg.contact_sensors[name])    # доступ: self.scene.sensors["robot0_ffdistal"]

# =====================================================================================================

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions[:, 0] += 30*np.pi/180
        actions[:, 5] -= 50*np.pi/180
        self.actions = self.action_scale * actions.clone()

        # self._update_and_apply_disturbances()

# =====================================================================================================

    def _apply_action(self):
        # --- контакт/геометрия для фазовой машины
        fforce, flags, fpose = self._read_contacts(threshold=self.contact_treshold, frame="w")  # силы и флаги в world frame

        com_w = self.object.data.body_com_pos_w[:, 0, 0:3]
        com   = com_w - self.env_origins

        table_pos = self.table.data.root_pos_w[:, :] - self.env_origins
        table_top_z = table_pos[:, 2] + 0.05

        # --- обновляем фазу (подъём UR10 скриптом) 
        self._update_phase_and_scripted_arm(flags, fpose, com, table_top_z)

        # print(self.actions*180/np.pi)

        # === ФАЗА 0: агент двигает пальцы ИНКРЕМЕНТАЛЬНО (со сглаживанием таргетов)
        if (self.phase == 0).any():
            ids = (self.phase == 0).nonzero(as_tuple=False).squeeze(-1)

            self.hand.set_joint_position_target(
                self.actions[ids], joint_ids=self.hand_joint_ids, env_ids=ids
            )

        # === ФАЗА 1: пальцы «заморожены» (держим хват)
        if (self.phase == 1).any():
            ids = (self.phase == 1).nonzero(as_tuple=False).squeeze(-1)
            # self.hand.set_joint_position_target(
            #     self.saved_grasp_targets[ids], joint_ids=self.hand_joint_ids, env_ids=ids
            # )
            self.hand.set_joint_position_target(
                self.actions[ids], joint_ids=self.hand_joint_ids, env_ids=ids
            )

        # self.hand.set_joint_position_target(
        #         self.actions, joint_ids=self.hand_joint_ids
        #     )

# =====================================================================================================

    def _get_observations(self):

        # self.cam.update(self.step_dt)
        # depth = self.cam.data.output["depth"]         # shape: (N, H, W), float32, метры
        # print(depth)
        # depth_0 = depth[0].cpu().numpy()              # (H, W)

        fforce, flags, fpose = self._read_contacts(threshold=self.contact_treshold, frame="w")  # силы и флаги в world frame

        object_pos = self.object.data.body_link_pose_w[:,0,:3]
        object_pos -= self.env_origins
        object_quat = self.object.data.body_link_pose_w[:,0,3:]

        # table_pos = self.table.data.root_pos_w[:, :] - self.env_origins
        # table_top_z = table_pos[:, 2] + 0.05
        # height = (object_pos[:, 2] - table_top_z).unsqueeze(-1)

        # ---- центр хвата (grasp center) из кончиков
        w = torch.clamp(flags, min=0.1)                    # (N,5)
        wsum = w.sum(dim=1, keepdim=True)                  # (N,1)
        grasp_center = (fpose * w.unsqueeze(-1)).sum(dim=1) / wsum  # (N,3)

        # относительный вектор от grasp_center до COM объекта
        rel_vec   = object_pos - grasp_center                     # (N,3)
        rel_dist  = torch.linalg.norm(rel_vec, dim=-1, keepdim=True)  # (N,1)
        rel_dir   = rel_vec / (rel_dist + 1e-6)            # (N,3)

        hand_open = self._hand_open_fraction().unsqueeze(1)

        obs = {
            "joints_pos": self.joint_pos,
            "joints_vel": self.joint_vel,

            "contact_forces": fforce,
            "contact_flags": flags,
            # "finger_tips_pos": fpose,

            "object_pos": object_pos,
            "object_quat": object_quat,

            "rel_dist": rel_dist,
            "rel_dir": rel_dir,

            "hand_open": hand_open,
        }

        return {"state": obs}
    
# =====================================================================================================

    def _get_rewards(self) -> torch.Tensor:

        fforce, flags, fpose = self._read_contacts(threshold=self.contact_treshold, frame="w")  # силы и флаги в world frame
   
        prev_flags = self.prev_contact_flags       # (N,5)

        object_com_w = self.object.data.body_com_pos_w[:, 0, 0:3]
        object_com   = object_com_w - self.env_origins

        # центр хвата и расстояние
        w = torch.clamp(flags, min=0.1)
        wsum = w.sum(dim=1, keepdim=True)
        grasp_center = (fpose * w.unsqueeze(-1)).sum(dim=1) / wsum
        rel_vec  = object_com - grasp_center
        rel_dist = torch.linalg.norm(rel_vec, dim=-1)         # (N,)

        # 1) тянуться к объекту
        # r_approach = 3.0 * (self._prev_rel_dist - rel_dist)
        r_approach = -1.0 * rel_dist

        # 2) пред-контакт: поощряем закрытие пальцев (только если контактов ещё нет)
        hand_open = self._hand_open_fraction()                # (N,)
        no_contact = (flags.sum(dim=1) == 0)
        delta_close = (self.prev_hand_open - hand_open).clamp(min=0.0)
        r_preclose = torch.where(no_contact, 2.0 * delta_close, torch.zeros_like(delta_close))

        # 3) новое касание (событие)
        # new_contacts = ((flags > 0.5) & (prev_flags <= 0.5)).float().sum(dim=1)
        # r_contact_event = 2.5 * new_contacts

        # 4) удержание контактов
        active_cnt = (flags > 0.5).float().sum(dim=1)
        r_contact_hold = 1.5 * active_cnt

        # 5) «распор»
        # fc = self._force_closure_score(flags, ft_pos, com, min_active=3, angle_deg=110.0)
        # r_fc = 1.8 * fc

        # flip_cnt = ((flags - prev_flags).abs() > 0).float().sum(dim=1)
        # r_stability = -0.2 * flip_cnt

        # === добавки за высоту
        table_pos = self.table.data.root_pos_w[:, :] - self.env_origins
        table_top_z = table_pos[:, 2] + 0.05
        obj_h = (object_com[:, 2] - table_top_z)
        r_h = 10.0 * obj_h

        # 6) небольшой штраф за время
        time_penalty = -0.01 * torch.ones(self.num_envs, device=self.device)

        rew = r_approach + r_preclose + r_contact_hold + r_h + time_penalty



        success_mask = (self.phase == 1) & (self.phase_step >= self.cfg.lift_steps) & (obj_h > self.cfg.success_height)
        fail_slip_mask = (self.phase == 1) & (self.slip_counter >= self.cfg.slip_grace)

        # падение по наклону
        obj_quat_w = self.object.data.root_quat_w
        local_z = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=obj_quat_w.dtype).expand(obj_quat_w.shape[0], 3)
        obj_z_axis_world = quat_apply(obj_quat_w, local_z)
        cos_tilt = obj_z_axis_world[:, 2].clamp(-1.0, 1.0)
        tilt_rad = torch.arccos(cos_tilt)
        fail_tilt_mask = tilt_rad > (self.cfg.tilt_fail_deg * torch.pi / 180.0)

        fail_mask = fail_slip_mask | fail_tilt_mask

        rew = rew + 5.0 * success_mask.float() - 1.0 * fail_mask.float()

        # === ВАЖНО: только теперь обновляем "предыдущие" величины
        with torch.no_grad():
            self.prev_contact_flags.copy_(flags)
            self.prev_hand_open.copy_(hand_open)

        return rew
    
# =====================================================================================================

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Определяем завершение эпизода по двум причинам:
        1) Превышение максимальной длины эпизода (truncation)
        2) Объект наклонён относительно мирового вертикаля сильнее порога cfg.tilt_fail_deg (termination)


        Вычисление наклона:
        - берём ориентацию объекта (кватернион world)
        - применяем его к локальному осевому вектору (0, 0, 1), чтобы получить направление оси Z объекта в мире
        - угол между этим направлением и мировым e_z даёт наклон
        """
        # ориентация первого (root) линка объекта во всех средах, форма (N, 4)
        obj_quat_w = self.object.data.body_link_pose_w[:, 0, 3:]

        # Локальная ось z объекта в мировых координатах через quat_apply
        local_z = torch.tensor([0.0, 0.0, 1.0],
                            device=self.device,
                            dtype=obj_quat_w.dtype).expand(obj_quat_w.shape[0], 3)
        
        # ось Z объекта в мировых координатах, форма (N, 3)
        obj_z_world = quat_apply(obj_quat_w, local_z)

        # косинус угла между obj_z_world и мировым e_z
        cos_tilt = torch.clamp((obj_z_world * local_z).sum(dim=-1), -1.0, 1.0)
        tilt_rad = torch.acos(cos_tilt)
        tilt_deg = tilt_rad * (180.0 / torch.pi)


        # termination по наклону
        terminations = tilt_deg > self.cfg.tilt_fail_deg
        # truncation по длине эпизода
        truncations = self.episode_length_buf >= self.max_episode_length


        return terminations, truncations
    
# =====================================================================================================

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)

        # joint_pos = torch.zeros((len(env_ids), 7), device=self.device)
        # joint_vel = torch.zeros((len(env_ids), 7), device=self.device)

        joint_pos = torch.zeros((len(env_ids), 26), device=self.device)
        joint_vel = torch.zeros((len(env_ids), 26), device=self.device)

        joint_pos[:, :6] = torch.tensor(self.cfg.start_position, device=self.device)*np.pi/180
        joint_pos[:, 6:] = torch.tensor(self.cfg.hand_position, device=self.device)*np.pi/180
 
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.hand.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.phase[env_ids] = 0
        self.phase_step[env_ids] = 0
        self.lift_hold_ok[env_ids] = 0
        self.grasp_ok_counter[env_ids] = 0
        self.slip_counter[env_ids] = 0

        with torch.no_grad():
            self.arm_start[env_ids] = self.joint_pos[env_ids][:, self.arm_joint_ids]
            delta = torch.tensor(self.cfg.lift_delta_deg, device=self.device) * torch.pi/180.0
            self.arm_goal[env_ids]  = self.arm_start[env_ids] + delta

        self.hand.set_joint_position_target(self.arm_start[env_ids], joint_ids=self.arm_joint_ids, env_ids=env_ids)

        self._reset_object_on_table(env_ids)

    # =====================================================================================================

    def _reset_object_on_table(self, env_ids):

        root_state = self.object.data.default_root_state[env_ids].clone()

        noise_xy = 0.02 * (torch.rand((len(env_ids), 2), device=self.device) - 0.5)
        root_state[:, 0] += noise_xy[:, 0]              # x
        root_state[:, 1] += noise_xy[:, 1]              # y

        # скорости в ноль
        root_state[:, 7:13] = 0.0

        # добавить сдвиг до центров копий среды
        root_state[:, 0:3] += self.scene.env_origins[env_ids]

        # записать в физику
        self.object.write_root_state_to_sim(root_state, env_ids)

# ===========================================================================
# ===========================================================================
# ===========================================================================

    def _read_contact_forces(self, frame: str = "w") -> torch.Tensor:
        """
        Возвращает тензор сил контакта размера (num_envs, num_fingers, 3).
        frame: 'w' — в мировых координатах, 'b' — в body frame (локально для линка).
        """
        assert frame in ("w", "b")
        forces_per_finger = []
        attr = f"net_forces_{frame}"

        for name in self.cfg.ft_names:
            s = self.scene.sensors[name]
            f = getattr(s.data, attr)  # shape: (N, 3) или (N, H, 3) при history_length>1
            if f.ndim == 3:
                f = f[:, -1, :]  # берём последний сэмпл истории

            fnorm = torch.linalg.norm(f, dim=-1)
            forces_per_finger.append(fnorm)

        return torch.stack(forces_per_finger, dim=1)  # (N, 5)
    
    # =====================================================================================================

    def _read_contact_flags(
        self,
        threshold: float = 1.0,
        frame: str = "w",
    ) -> torch.Tensor:
        """
        Булевы флаги контакта (norm(force) > threshold) размера (num_envs, num_fingers).
        threshold — порог в Ньютонах.
        """
        F = self._read_contact_forces(frame=frame)          # (N, 5, 3)
        norms = torch.linalg.norm(F, dim=-1)                # (N, 5)
        return norms > threshold
    
    # =====================================================================================================
    
    def compute_force_closure(env, contact_sensors, object_pose, friction_coeff=0.8, cone_sides=8):
        """
        Проверка force closure для контактов с объектом.
        
        Args:
            env: среда Isaac Lab
            contact_sensors: список ContactSensorCfg (по одному на палец)
            object_pose: (pos, quat) объекта
            friction_coeff: μ для конуса трения
            cone_sides: аппроксимация конуса (больше → точнее)
        Returns:
            bool: True если force closure, False иначе
        """
        # центр объекта
        obj_pos = object_pose[0]

        wrenches = []

        for sensor in contact_sensors:
            contacts = sensor.data.body_contact_forces  # [N_contacts, 3]
            points = sensor.data.body_contact_positions # [N_contacts, 3]
            
            for f, p in zip(contacts, points):
                if torch.norm(f) < 1e-6:
                    continue  # нет контакта
                
                n = f / (torch.norm(f) + 1e-8)  # нормаль контакта
                # генерируем "пирамиду" трения
                basis = torch.randn(3, cone_sides)
                basis = torch.nn.functional.normalize(basis, dim=0)
                for i in range(cone_sides):
                    # сила внутри конуса
                    fi = n + friction_coeff * basis[:, i]
                    fi = fi / torch.norm(fi) * torch.norm(f)
                    # момент = r × f
                    r = p - obj_pos
                    tau = torch.cross(r, fi)
                    wrench = torch.cat([fi, tau])
                    wrenches.append(wrench)
        
        if len(wrenches) < 6:
            return False
        
        W = torch.stack(wrenches, dim=1)  # [6, M]
        # Проверяем ранг
        if torch.linalg.matrix_rank(W) < 6:
            return False
        
        # TODO: Можно добавить LP-проверку, что ноль лежит внутри конуса
        return True
    
    # =====================================================================================================        

    def _read_contacts(
        self,
        threshold: float = 1.0,
        frame: str = "w",
    ):
        """
        Удобный комбинированный вызов: возвращает (forces, flags)
        forces: (N, 5, 3), flags: (N, 5)
        """
        F = self._read_contact_forces(frame=frame)
        flags = (F > threshold).float()

        ft_pos_w = self.hand.data.body_pos_w[:, self.ft_ids, :]         # (N,5,3) world
        ft_pos   = ft_pos_w - self.env_origins.unsqueeze(1)

        return F, flags, ft_pos
    
# ===========================================================================
# ===========================================================================
# ===========================================================================

    def _update_phase_and_scripted_arm(self, flags, ft_pos, com_local, table_top_z):
        """
        Управляет переходами фаз и скриптовым подъёмом UR10.
        """
        
        # --- критерий хват-фазы (фаза 0)
        # 1) достаточно активных пальцев, 2) force-closure высокий, 3) COM близко к центру хвата

        active_cnt = (flags > 0.5).sum(dim=1)

        grasp_condition = (active_cnt >= self.cfg.grasp_contact_min)

        # считаем последовательные шаги с выполненным условием
        self.grasp_ok_counter = torch.where(
                                    grasp_condition,
                                    self.grasp_ok_counter + 1,
                                    torch.clamp(self.grasp_ok_counter - 2, min=0)
                                )
        # --- переход 0 -> 1
        start_lift_mask = (self.phase == 0) & (self.grasp_ok_counter >= self.cfg.grasp_hold_steps)
        if start_lift_mask.any():
            ids = start_lift_mask.nonzero(as_tuple=False).squeeze(-1)
            # фиксируем цели пальцев (замораживаем хват)
            self.saved_grasp_targets[ids] = self.hand.data.joint_pos[ids][:, self.hand_joint_ids]
            # перезапускаем счетчики
            self.phase[ids] = 1
            self.phase_step[ids] = 0
            self.slip_counter[ids] = 0
            self.lift_hold_ok[ids] = 0
            # уже записаны arm_start/arm_goal в reset; на подъем просто интерполируем

        # --- если мы в фазе 1 — двигаем руку по линейной интерполяции и следим за «скольжением»
        in_lift = (self.phase == 1)
        if self.scripted_lift_enabled and in_lift.any():
            ids = in_lift.nonzero(as_tuple=False).squeeze(-1)
            t = (self.phase_step[ids].float() / max(1, self.cfg.lift_steps)).clamp(0, 1).unsqueeze(1)  # (B,1)
            targ = (1 - t) * self.arm_start[ids] + t * self.arm_goal[ids]                              # (B, n_arm)

            self.hand.set_joint_position_target(
                targ, joint_ids=self.arm_joint_ids, env_ids=ids
            )

            # критерий «хват не потерян»: либо контакты держатся, либо объект реально поднят
            obj_height = (com_local[ids, 2] - table_top_z[ids])  # в метрах
            good = (active_cnt[ids] >= self.cfg.grasp_contact_min) | (obj_height > self.cfg.success_height)
            self.slip_counter[ids] = torch.where(good, torch.zeros_like(self.slip_counter[ids]),
                                                self.slip_counter[ids] + 1)

            self.phase_step[ids] += 1
    
    # =====================================================================================================        
    
    def _force_closure_score(self, flags, ft_pos, com, min_active=2, angle_deg=110.0):
        """
        flags: (N,5)  — какие пальцы активны
        ft_pos:(N,5,3)
        com:   (N,3)
        Возвращает скалярный score (N,), 0..1.
        """
        N = flags.shape[0]
        score = torch.zeros(N, device=self.device)

        # направления (N,5,3), нормированные
        vec = ft_pos - com.unsqueeze(1)                    # к пальцам
        vec = vec / (torch.norm(vec, dim=-1, keepdim=True) + 1e-8)

        # маска только активных пальцев
        mask = flags > 0.5                                 # (N,5)

        # для каждой среды посчитаем max попарный угол между активными
        cos_thr = torch.cos(torch.tensor(angle_deg/180.0*math.pi, device=self.device))
        for i in range(N):
            idx = mask[i].nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() < min_active:
                score[i] = 0.0
                continue
            v = vec[i, idx, :]                             # (k,3)
            # попарные косинусы углов
            # cos = v_a · v_b; большой противоположный угол => cos ~ -1
            cos_mat = (v @ v.t()).clamp(-1.0, 1.0)         # (k,k)
            # игнор диагональ
            cos_mat.fill_diagonal_(1.0)
            min_cos = cos_mat.min()                        # самый "противоположный" косинус
            # если есть пара с cos < cos_thr (т.е. угол > angle_deg), считаем OK
            s = torch.sigmoid(10.0*(cos_thr - min_cos))    # плавный переход
            score[i] = s

        return score  # (N,), 0..1

    # =====================================================================================================        

    def _hand_open_fraction(self):
        """
        Грубая метрика "насколько рука открыта" в [0..1], где 1 = максимально раскрыта.
        Используем только сгибающие DOF пальцев, без запястий.
        """

        q = self.joint_pos[:, self.hand_joint_ids]                 # (N, n_finger_dof)
        # нормируем по диапазону: считаем, что 0 ~ открыта, положительные ~ закрытие (у тебя init=0)
        # на практике лучше считать по реальным joint_limits, но пока — масштабирующий tanh:
        open_frac = torch.sigmoid(-3.0 * q).mean(dim=1)   # чем больше q (закрытие), тем меньше open_frac
        return open_frac.clamp(0.0, 1.0)

