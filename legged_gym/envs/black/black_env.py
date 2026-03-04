
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class BlackRobot(LeggedRobot):
    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions

        return noise_vec
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((self.base_ang_vel  * self.obs_scales.ang_vel,
                                  self.projected_gravity,
                                  self.commands[:, :3] * self.commands_scale,
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                  self.dof_vel * self.obs_scales.dof_vel,
                                  self.actions,
                                  ),dim=-1)
        
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.0) * self.obs_scales.height_measurements
        
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) * 1e-3,  # foot contact forces (4,)
                                    self.torques / self.torque_limits,  # motor torques (12,)
                                    (self.last_dof_vel - self.dof_vel) / self.dt * 1e-4,  # motor accelerations (12,)
                                    heights,  # height measurements (187,)
                                    ),dim=-1)
        # print(f"foot contact: {self.privileged_obs_buf[:,48:48+4].min(), self.privileged_obs_buf[:,48:48+4].max()}")
        # print(f"torques: {self.privileged_obs_buf[:,48+4:48+4+12].min(), self.privileged_obs_buf[:,48+4:48+4+12].max()}")
        # print(f"acc: {self.privileged_obs_buf[:,48+4+12:48+4+12+12].min(), self.privileged_obs_buf[:,48+4+12:48+4+12+12].max()}")
        
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _reward_hip_to_default(self):
        hip_dof_names = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint']
        hip_dof_indices = [0, 3, 6, 9]
        hip_pos = self.dof_pos[:, hip_dof_indices]
        default_hip_pos = self.default_dof_pos[:, hip_dof_indices]
        rew = torch.sum(torch.abs(hip_pos - default_hip_pos), dim=1)

        dynamic_cfg = getattr(self.cfg.rewards, "hip_to_default_dynamic", None)
        if dynamic_cfg is None:
            return rew

        abs_lin_y = torch.abs(self.commands[:, 1])
        abs_yaw = torch.abs(self.commands[:, 2])

        lin_y_denom = max(dynamic_cfg["lin_vel_y_max"] - dynamic_cfg["lin_vel_y_threshold"], 1e-6)
        yaw_denom = max(dynamic_cfg["ang_vel_yaw_max"] - dynamic_cfg["ang_vel_yaw_threshold"], 1e-6)

        lin_y_ratio = torch.clamp((abs_lin_y - dynamic_cfg["lin_vel_y_threshold"]) / lin_y_denom, 0.0, 1.0)
        yaw_ratio = torch.clamp((abs_yaw - dynamic_cfg["ang_vel_yaw_threshold"]) / yaw_denom, 0.0, 1.0)

        # Strongest reduction appears when both lateral and yaw commands are active.
        activity = 0.5 * (lin_y_ratio + yaw_ratio)
        min_coef = float(dynamic_cfg["min_coef"])
        dynamic_coef = 1.0 - (1.0 - min_coef) * activity
        return rew * dynamic_coef

    def _reward_body_orientation(self):
        roll = self.rpy[:, 0]
        pitch = self.rpy[:, 1]
        deadzone_cfg = getattr(self.cfg.rewards, "body_orientation_deadzone", None)

        if deadzone_cfg is not None:
            roll_err = torch.clamp(torch.abs(roll) - float(deadzone_cfg["roll"]), min=0.0)
            pitch_err = torch.clamp(torch.abs(pitch) - float(deadzone_cfg["pitch"]), min=0.0)
        else:
            roll_err = torch.abs(roll)
            pitch_err = torch.abs(pitch)

        rew = torch.square(roll_err) + torch.square(pitch_err)

        dynamic_cfg = getattr(self.cfg.rewards, "body_orientation_dynamic", None)
        if dynamic_cfg is None:
            return rew

        abs_lin_y = torch.abs(self.commands[:, 1])
        abs_yaw = torch.abs(self.commands[:, 2])

        lin_y_denom = max(dynamic_cfg["lin_vel_y_max"] - dynamic_cfg["lin_vel_y_threshold"], 1e-6)
        yaw_denom = max(dynamic_cfg["ang_vel_yaw_max"] - dynamic_cfg["ang_vel_yaw_threshold"], 1e-6)
        lin_y_ratio = torch.clamp((abs_lin_y - dynamic_cfg["lin_vel_y_threshold"]) / lin_y_denom, 0.0, 1.0)
        yaw_ratio = torch.clamp((abs_yaw - dynamic_cfg["ang_vel_yaw_threshold"]) / yaw_denom, 0.0, 1.0)

        # Estimate roughness from local height variance.
        roughness_ratio = torch.zeros_like(rew)
        if self.cfg.terrain.measure_heights:
            heights_std = torch.std(self.measured_heights, dim=1)
            rough_denom = max(dynamic_cfg["roughness_max"] - dynamic_cfg["roughness_threshold"], 1e-6)
            roughness_ratio = torch.clamp((heights_std - dynamic_cfg["roughness_threshold"]) / rough_denom, 0.0, 1.0)

        activity = torch.maximum(torch.maximum(lin_y_ratio, yaw_ratio), roughness_ratio)
        min_coef = float(dynamic_cfg["min_coef"])
        dynamic_coef = 1.0 - (1.0 - min_coef) * activity
        return rew * dynamic_coef

    def _reward_roll_bias(self):
        if not hasattr(self, "roll_ema"):
            self.roll_ema = torch.zeros(self.num_envs, device=self.device)

        # Clear history for environments that are about to reset.
        if hasattr(self, "reset_buf"):
            self.roll_ema[self.reset_buf] = 0.0

        alpha = float(getattr(self.cfg.rewards, "roll_bias_ema_alpha", 0.98))
        self.roll_ema = alpha * self.roll_ema + (1.0 - alpha) * self.rpy[:, 0]
        return torch.abs(self.roll_ema)

    def _reward_x_command_hip_regular(self):
        hip_dof_indices = [0, 3, 6, 9]
        hip_pos = self.dof_pos[:, hip_dof_indices]
        cmd_norm = torch.norm(self.commands[:, :3], dim=1)
        # Avoid NaNs when command norm is near zero.
        x_command_ratio = torch.where(
            cmd_norm > 1e-6,
            torch.abs(self.commands[:, 0]) / torch.clamp_min(cmd_norm, 1e-6),
            torch.zeros_like(cmd_norm),
        )
        rew = torch.abs(hip_pos[:,0]+hip_pos[:,1]) + torch.abs(hip_pos[:,2]+hip_pos[:,3])
        return rew * x_command_ratio
