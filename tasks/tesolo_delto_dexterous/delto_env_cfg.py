import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
# from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors import CameraCfg

from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

# from isaaclab.utils.math import quat_to_rot_mats

##
# Configuration
##
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .ur_delto_cfg import DELTO_CFG


# =====================================================================================================
# =====================================================================================================
# =====================================================================================================

@configclass
class DeltoEnvCfg(DirectRLEnvCfg):

    # ======================================================================= env params
    decimation = 2
    episode_length_s = 5.0
    action_space = 26
    observation_space = 121 * 5 # 63
    state_space = 0
    action_scale = [0.05] * 6 + [0.01] * 20
    # action_scale = [0.1] * 6 + [0.05] * 20
    asymmetric_obs = False
    num_env = 32

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

    # hand_position = [0.0, 0.0, 0.0, 0.0, 0.0,
    #                  0.0, 0.0, 0.0, 0.0, 0.0, 
    #                  0.0, 0.0, 0.0, 0.0, 0.0, 
    #                  0.0, 0.0, 0.0, 0.0, 0.0]
    
    hand_position = [30.0, 0.0, 0.0, 0.0, 0.0,
                     -50.0, 30.0, 30.0, 30.0, 0.0, 
                     0.0, 30.0, 30.0, 30.0, 30.0, 
                     0.0, 30.0, 30.0, 30.0, 30.0]
    
    arm_position = [-90.0, -90.0, 100.0, -10.0, 90.0, 0.0]

    hand_upper_limits = [70, 31, 30, 15, 60,
                        0, 115, 115, 110, 90,
                        90, 90, 90, 90, 90,
                        90, 90, 90, 90, 90]
    
    hand_lower_limits = [-22, -20, -30, -32, 0,
                        -155, 0, 0, 0, -15,
                        -90, -90, -90, -90, -90,
                        -90, -90, -90, -90, -90]

    arm_upper_limits = [360, 360, 180, 360, 360, 360]
    arm_lower_limits = [-360, -360, -180, -360, -360, -360]

    # ======================================================================= sensors

    ft_names = [
        "rl_dg_1_tip",
        "rl_dg_2_tip",
        "rl_dg_3_tip",
        "rl_dg_4_tip",
        "rl_dg_5_tip",
        # "rl_dg_1_4",
        # "rl_dg_2_4",
        # "rl_dg_3_4",
        # "rl_dg_4_4",
        # "rl_dg_5_4",
    ]

    contact_sensors = {}
    for name in ft_names:
        contact_sensors[name] = ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/dg5f_my/{name}",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=["/World/envs/env_.*/Object"],
            track_air_time=False,
        )

    # ======================================================================= objects

    # in-manipulator object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",  
        spawn=sim_utils.CylinderCfg(
            radius=0.04,
            height=0.1,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=2.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),
                metallic=0.1,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.83, 0.1 + 0.05),
            rot=(0.7071, 0, 0, -0.7071)
            # pos=(0.22, -1.0, 0.1),
        ),
    )

    table_cfg: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Table",
            spawn=sim_utils.CuboidCfg( 
                size=(1.0, 0.5, 0.1),
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
                pos=(0.0, -0.85, 0.05),
            ),
        )
    
    ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/ObservationPointCloud")
    ray_cfg.markers["hit"].radius = 0.0025
    
    # ======================================================================= scene

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32, 
        env_spacing= 3.0, 
        replicate_physics=True
        )
    

    grasp_contact_min = 3            # минимум активных пальцев
    success_height = 0.40            # м: на столько поднять выше поверхности стола

    #  ======================================================================= rewards