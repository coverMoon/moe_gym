import math
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO, LeggedRobotCfgCTS, LeggedRobotCfgMoENGCTS, LeggedRobotCfgMoENGCTS, LeggedRobotCfgMCPCTS, LeggedRobotCfgACMoECTS, LeggedRobotCfgDualMoECTS, LeggedRobotCfgMoECTS

class BLACKCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        # 1. 初始姿态
        pos = [0.0, 0.0, 0.45] 
        default_joint_angles = {
            'FL_hip_joint': 0.0,   'FL_thigh_joint': 0.8014,   'FL_calf_joint': -1.527,
            'FR_hip_joint': -0.0,  'FR_thigh_joint': -0.8014,  'FR_calf_joint': 1.527,
            'RL_hip_joint': 0.0,   'RL_thigh_joint': 0.8014,   'RL_calf_joint': -1.527,
            'RR_hip_joint': -0.0,  'RR_thigh_joint': -0.8014,  'RR_calf_joint': 1.527
        }
        
        turn_over = False # initialize the robot in a flipped over position
        # turn_over_proportions = [0.1, 0.3, 0.6] # proportions for backflip, sideflip, noflip
        turn_over_proportions = [0.0, 0.2, 0.8] # proportions for backflip, sideflip, noflip
        turn_over_init_heights = { # initial heights range for each flip type
            'backflip': [0.10, 0.15],
            'sideflip': [0.16, 0.21],
        }
        # turn_over_proportions = [0.0, 1.0, 0.0] # proportions for backflip, sideflip, noflip

    class env(LeggedRobotCfg.env):
        num_envs = 8192
        num_observations = 45
        # obs(45) + base_lin_vel(3) + height_measurements(187)
        num_privileged_obs = 45 + 3 + 4 + 12 + 12 + 187  # 263
        # num_privileged_obs = 45 + 3 + 187  # 235
        # num_privileged_obs = 48  # without height measurements
        episode_length_s = 25

    class domain_rand(LeggedRobotCfg.domain_rand):
        ### Robot properties ###
        randomize_friction = True
        friction_range = [0.0, 2.0]

        randomize_base_mass = True
        added_mass_range = [-1., 1.]

        randomize_link_mass = True
        multiplied_link_mass_range = [0.75, 1.25]

        randomize_base_com = True
        added_base_com_range = [-0.05, 0.05]

        randomize_restitution = True # restitution to robot links (Robot init)
        restitution_range = [0.0, 0.5]

        ### Environment reset ###
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  
        damping_multiplier_range = [0.8, 1.2]    

        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035]

        randomize_motor_strength = True # (Env reset)
        motor_strength_range = [0.8, 1.2]

        ### Environment step ###
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6

        randomize_action_delay = True # use last_action with 0~20 ms delay, 4 decimation

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        # 刚度 (P Gain)
        stiffness = {
            'FL_hip_joint': 40.0, 'RL_hip_joint': 40.0, 'FR_hip_joint': 40.0, 'RR_hip_joint': 40.0,
            'FL_thigh_joint': 40.0, 'RL_thigh_joint': 40.0, 'FR_thigh_joint': 40.0, 'RR_thigh_joint': 40.0,
            'FL_calf_joint': 40.0, 'RL_calf_joint': 40.0, 'FR_calf_joint': 40.0, 'RR_calf_joint': 40.0
        }
        # 阻尼 (D Gain)
        damping = {
            'FL_hip_joint': 1.2, 'RL_hip_joint': 1.2, 'FR_hip_joint': 1.2, 'RR_hip_joint': 1.2,
            'FL_thigh_joint': 1.2, 'RL_thigh_joint': 1.0, 'FR_thigh_joint': 1.2, 'RR_thigh_joint': 1.2,
            'FL_calf_joint': 1.2, 'RL_calf_joint': 1.2, 'FR_calf_joint': 1.2, 'RR_calf_joint': 1.2
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh' # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        terrain_spacing = 0.5 # spacing between different terrain types [m]

        # [wave, slope, rough_slope, stairs up, stairs down, obstacles, stepping_stones, gap, flat, high_wall]
        # terrain_proportions = [0.2, 0.05, 0.05, 0.30, 0.05, 0.25, 0.0, 0.0, 0.1]  # 更偏向wave
        terrain_proportions = [0.05, 0.20, 0.05, 0.25, 0.10, 0.15, 0.0, 0.0, 0.10, 0.10]  # 加入 high_wall(30cm, 10cm)
        # terrain_proportions = [0.20, 0.05, 0.05, 0.30, 0.15, 0.20, 0.0, 0.0, 0.05]  # 更偏向wave和stairs
        # terrain_proportions = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        # terrain_proportions = [0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
        # terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
        move_down_by_accumulated_xy_command = True # move down the terrain curriculum based on accumulated xy command distance instead of absolute distance
        
    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        # start training with zero commands and then gradually increase zero command probability
        zero_command_curriculum = {'start_iter': 0, 'end_iter': 1500, 'start_value': 0.0, 'end_value': 0.1}
        limit_ang_vel_at_zero_command_prob = 0.2 # probability of add limiting angular velocity commands when zero command is sampled
        limit_vel_prob = 0.2 # probability of limiting linear velocity command
        limit_vel_invert_when_continuous = True # invert the limit logic when using continuous sample limit velocity commands
        limit_vel = {"lin_vel_x": [-1, 1], "lin_vel_y": [-1, 1], "ang_vel_yaw": [-1, 0, 1]} # sample vel commands from min [-1] or zero [0] or max [1] range only
        stop_heading_at_limit = True # stop heading updates when vel is limited
        dynamic_resample_commands = True # sample commands with low bounds
        command_range_curriculum = [{ # list for command range curriculums at specific training iterations
            'iter': 20000, # training iteration at which the command ranges are updated
            'lin_vel_x': [-1.0, 1.0], # min max [m/s]
            'lin_vel_y': [-1.0, 1.0], # min max [m/s]
            'ang_vel_yaw': [-1.5, 1.5], # min max [rad/s]
            'heading': [-1.57, 1.57], # min max [rad]
        }, { # list for command range curriculums at specific training iterations
            'iter': 50000, # training iteration at which the command ranges are updated
            'lin_vel_x': [-2.0, 2.0], # min max [m/s]
            'lin_vel_y': [-1.0, 1.0], # min max [m/s]
            'ang_vel_yaw': [-2.0, 2.0], # min max [rad/s]
            'heading': [-1.57, 1.57], # min max [rad]
        }]
        turn_over_zero_time = { # if turn_over is true, time robot must be stable before sampling new commands after a turn over
            "backflip": 5.0,
            "sideflip": 3.0,
        }
        # [wave, slope, rough slope, stairs up, stairs down, obstacles, stepping stones, gap, flat, high_wall]
        terrain_max_command_ranges = [
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # wave
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # slope
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # rough slope
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs up
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs down
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # obstacles
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stepping stones
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # gap
            {'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0], 'heading': [-1.57, 1.57]},  # flat
            {'lin_vel_x': [-0.8, 0.8], 'lin_vel_y': [-0.6, 0.6], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [-1.57, 1.57]},  # high_wall
        ]

        class ranges:
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.5, 0.5] # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]   # min max [rad/s]
            heading = [-1.57, 1.57] # min max [rad]
        
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/black/urdf/black_description.urdf'
        name = "black"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.43
        only_positive_rewards = False
        max_contact_force = 147. # forces above this value are penalized, go2 weight 15kg
        curriculum_rewards = [
            {'reward_name': 'lin_vel_z', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0},
            {'reward_name': 'correct_base_height', 'start_iter': 0, 'end_iter': 5000, 'start_value': 1.0, 'end_value': 10.0},
            # {'reward_name': 'dof_power', 'start_iter': 0, 'end_iter': 3000, 'start_value': 1.0, 'end_value': 0.1},
            # {'reward_name': 'upright', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0},
        ]
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        dynamic_sigma = { # linear interpolation of sigma based on command velocity, **Must start terrain curriculum first**
            "min_lin_vel": 0.5, # min abs linear velocity to have default sigma
            "max_lin_vel": 1.5, # max abs linear velocity to have max sigma
            "min_ang_vel": 1.0, # min abs angular velocity to have default sigma
            "max_ang_vel": 2.0, # max abs angular velocity to have max sigma
            # wave, slope, rough_slope, stairs up, stairs down, obstacles, stepping_stones, gap, flat, high_wall]
            # "max_sigma": [1/3, 1/4, 1/4, 1/2.7, 1/2.7, 1/2, 1, 1, 1/4]
            "max_sigma": [5/12, 1/4, 1/4, 1/2, 1/2, 3/4, 1, 1, 1/4, 1/2]
        }
        min_legs_distance = 0.1  # min distance between legs to not be considered stumbling
        # Dynamic coefficient for hip_to_default reward:
        # reduce penalty when lateral/yaw commands are active.
        hip_to_default_dynamic = {
            "min_coef": 0.2,            # minimum multiplier at high lateral+yaw command
            "lin_vel_y_threshold": 0.1,  # start reducing beyond this |lin_vel_y|
            "lin_vel_y_max": 1.0,        # full lateral contribution at this |lin_vel_y|
            "ang_vel_yaw_threshold": 0.2,# start reducing beyond this |ang_vel_yaw|
            "ang_vel_yaw_max": 3.0,      # full yaw contribution at this |ang_vel_yaw|
        }
        # Deadzone around level pose so tiny tilt does not get over-penalized.
        body_orientation_deadzone = {
            "roll": 0.05,   # rad
            "pitch": 0.10,  # rad
        }
        # Dynamic coefficient for body orientation penalty.
        # Reduce penalty under side/yaw commands and rough terrain.
        body_orientation_dynamic = {
            "min_coef": 0.50,
            "lin_vel_y_threshold": 0.10,
            "lin_vel_y_max": 1.00,
            "ang_vel_yaw_threshold": 0.20,
            "ang_vel_yaw_max": 3.00,
            "roughness_threshold": 0.02,
            "roughness_max": 0.10,
        }
        # Persistent roll-bias penalty (EMA roll).
        roll_bias_ema_alpha = 0.98
        class scales:
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_acc = -2.5e-7
            dof_power = -2e-5
            torques = -1e-4
            correct_base_height = -1.0
            action_rate = -0.015
            action_smoothness = -0.02
            stand_still = -1.0
            body_orientation = -1.5
            roll_bias = -2.0
            collision = -1.0
            dof_pos_limits = -2.0
            feet_regulation = -0.05
            x_command_hip_regular = -0.15
            # CTS reward trains to have very close feet distance, real robot performance is poor, but sim2sim can climb 20cm stairs, try to add hip_to_default reward or similar_to_default reward
            # training to y=1.5, font feet will collide noticeably, max y=1.0
            hip_to_default = -0.12
            # legs_distance = -1.5  # not good performance, avoid leg collision
            # similar_to_default = -0.01
            # feet_contact_forces = -1.0  # try to add but no effect, remove

        turn_over_roll_threshold = math.pi / 4 # threshold on roll to use turn over rewards
        class turn_over_scales:
            upright = 1.0
            # dof_acc = -2.5e-7
            # dof_power = -2e-5
            # action_rate = -0.001
            # action_smoothness = -0.001

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class BLACKCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'black_ppo'
        max_iterations = 150000
        save_interval = 500

class BLACKCfgCTS(LeggedRobotCfgCTS):
    class runner(LeggedRobotCfgCTS.runner):
        num_steps_per_env = 24
        run_name = ''
        experiment_name = 'black_cts'
        max_iterations = 150000
        save_interval = 500
    
    class policy(LeggedRobotCfgCTS.policy):
        latent_dim = 32
        norm_type = 'l2norm'

class BLACKCfgMoENGCTS(LeggedRobotCfgMoENGCTS):
    class policy(LeggedRobotCfgMoENGCTS.policy):
        obs_no_goal_mask = [True] * 6 + [False] * 3 + [True] * 36  # mask for obs without command info
        student_expert_num = 8 # number of experts in the student model
    
    class algorithm(LeggedRobotCfgMoENGCTS.algorithm):
        load_balance_coef = 0.01

    class runner(LeggedRobotCfgMoENGCTS.runner):
        run_name = ''
        experiment_name = 'black_moe_no_goal_cts'
        max_iterations = 150000
        save_interval = 500

class BLACKCfgMCPCTS(LeggedRobotCfgMCPCTS):
    class policy(LeggedRobotCfgMCPCTS.policy):
        obs_no_goal_mask = [True] * 6 + [False] * 3 + [True] * 36  # mask for obs without command info
        student_expert_num = 8 # number of experts in the student model

    class runner(LeggedRobotCfgMCPCTS.runner):
        run_name = ''
        experiment_name = 'black_mcp_cts'
        max_iterations = 150000
        save_interval = 500

class BLACKCfgACMoECTS(LeggedRobotCfgACMoECTS):
    class policy(LeggedRobotCfgACMoECTS.policy):
        expert_num = 8  # number of experts in the student model
    
    class runner(LeggedRobotCfgACMoECTS.runner):
        run_name = ''
        experiment_name = 'black_ac_moe_cts'
        max_iterations = 150000
        save_interval = 500

class BLACKCfgDualMoECTS(LeggedRobotCfgDualMoECTS):
    class policy(LeggedRobotCfgDualMoECTS.policy):
        expert_num = 8  # number of experts in the student model
    
    class runner(LeggedRobotCfgDualMoECTS.runner):
        run_name = ''
        experiment_name = 'black_dual_moe_cts'
        max_iterations = 150000
        save_interval = 500

class BLACKCfgMoECTS(LeggedRobotCfgMoECTS):
    class policy(LeggedRobotCfgMoECTS.policy):
        expert_num = 8  # number of experts in the student model
    
    class runner(LeggedRobotCfgMoECTS.runner):
        run_name = ''
        experiment_name = 'black_moe_cts'
        max_iterations = 150000
        save_interval = 500
