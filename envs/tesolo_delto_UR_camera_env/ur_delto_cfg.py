from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg




DELTO_CFG = ArticulationCfg(
    # prim_path = "/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"robots/dg5f_right/ur10e_delto_camera.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            # max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            # sleep_threshold=0.005,
            # stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(
        #     collision_enabled=True,
        #     contact_offset=0.002,  # отступ для генерации контактов
        #     rest_offset=0.0,       # убрать "воздушный зазор"
        # ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={"shoulder_pan_joint": -90.0 *torch.pi/180,
                    "shoulder_lift_joint": -75.0 *torch.pi/180,
                    "elbow_joint": 120.0 *torch.pi/180,
                    "wrist_1_joint": -50.0 *torch.pi/180,
                    "wrist_2_joint": 90.0 *torch.pi/180,
                    "wrist_3_joint": -90.0 *torch.pi/180,
                    "rj_dg_.*": 0.0
                },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint",
                                "shoulder_lift_joint",
                                "elbow_joint",
                                "wrist_1_joint",
                                "wrist_2_joint",
                                "wrist_3_joint"],
            velocity_limit=1000.0,
            effort_limit=870.0,
            stiffness=800.0,
            damping=40.0,
        ),

        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["rj_dg_.*"],
            effort_limit={
                "rj_dg_.*":30
            },
            stiffness={
                "rj_dg_1_1": 0.8294,
                "rj_dg_1_2": 0.62859, 
                "rj_dg_1_3": 0.45859, 
                "rj_dg_1_4": 0.16977, 

                "rj_dg_2_1": 2.43808, 
                "rj_dg_2_2": 0.82762, 
                "rj_dg_2_3": 0.41117, 
                "rj_dg_2_4": 0.09574, 

                "rj_dg_3_1": 2.45355, 
                "rj_dg_3_2": 0.82643,
                "rj_dg_3_3": 0.41118, 
                "rj_dg_3_4": 0.09575, 

                "rj_dg_4_1": 2.3093,
                "rj_dg_4_2": 0.82729,
                "rj_dg_4_3": 0.41094,
                "rj_dg_4_4": 0.09573,

                "rj_dg_5_1": 1.73423,
                "rj_dg_5_2": 1.10747,
                "rj_dg_5_3": 0.45515,
                "rj_dg_5_4": 0.16936
            },
            damping={
         
                "rj_dg_1_1": 0.05314332,
                "rj_dg_1_2": 0.02990215, 
                "rj_dg_1_3": 0.01657904, 
                "rj_dg_1_4": 0.00270917, 

                "rj_dg_2_1": 0.26783944, 
                "rj_dg_2_2": 0.0451749, 
                "rj_dg_2_3": 0.0120614, 
                "rj_dg_2_4": 0.00088871, 

                "rj_dg_3_1": 0.2703927,
                "rj_dg_3_2": 0.04507756,
                "rj_dg_3_3": 0.0125066, 
                "rj_dg_3_4": 0.00088885, 

                "rj_dg_4_1": 0.24690116,
                "rj_dg_4_2": 0.04514794,
                "rj_dg_4_3": 0.01249565,
                "rj_dg_4_4": 0.00088857,

                "rj_dg_5_1": 0.16547778,
                "rj_dg_5_2": 0.0699276,
                "rj_dg_5_3": 0.01639284,
                "rj_dg_5_4": 0.0026993,

                # "rj_dg_1_1": 0.00033,
                # "rj_dg_1_2": 0.00025, 
                # "rj_dg_1_3": 0.00018, 
                # "rj_dg_1_4": 0.00007, 

                # "rj_dg_2_1": 0.00093, 
                # "rj_dg_2_2": 0.00033, 
                # "rj_dg_2_3": 0.00016, 
                # "rj_dg_2_4": 0.00004, 

                # "rj_dg_3_1": 0.00098, 
                # "rj_dg_3_2": 0.00033,
                # "rj_dg_3_3": 0.00016, 
                # "rj_dg_3_4": 0.00004, 

                # "rj_dg_4_1": 0.00092,
                # "rj_dg_4_2": 0.00033,
                # "rj_dg_4_3": 0.00016,
                # "rj_dg_4_4": 0.00004,

                # "rj_dg_5_1": 0.00069,
                # "rj_dg_5_2": 0.00044,
                # "rj_dg_5_3": 0.00018,
                # "rj_dg_5_4": 0.00007
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)