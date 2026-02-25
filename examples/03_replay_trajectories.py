import sys
import os
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt

import omnigibson as og

from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.helpers import flip_pose_pointing_down
from realm.inference import extract_from_obs
from realm.eval import set_sim_config


def replay_traj(trajectory_actions, trajectory_gt_qpos, trajectory_gt_ee):
    max_steps = min(len(trajectory_actions), 1000)

    qpos = []
    ee_pos_list = []

    obs, _ = env.reset()
    obs, rew, terminated, truncated, info = env.warmup(obs)

    for _ in range(150):
        action = trajectory_gt_qpos[0]
        obs, curr_task_progression, terminated, truncated, info = env.step(action)

    for t in range(max_steps):
        base_im, base_im_second, wrist_im, robot_state, gripper_state = extract_from_obs(obs)

        ee_pos, ee_rot = env.get_ee_pose()
        ee_pos_list.append(ee_pos)

        qpos.append(np.concatenate((robot_state, np.atleast_1d(np.array(gripper_state)))))

        action = trajectory_actions[t]

        obs, curr_task_progression, terminated, truncated, info = env.step(action)

    # Stack final achieve trajectories:
    qpos_arr = np.stack(qpos)  # (N, 8)
    qpos_joints = qpos_arr[:, :7]
    ee_pos_arr = np.stack(ee_pos_list)


    qpos_err= qpos_joints[:, :7] - trajectory_gt_qpos[:, :7]
    ee_pos_err = ee_pos_arr[:, :] - trajectory_gt_ee[:, :3]

    return {
        "qpos_err": qpos_err,
        "ee_pos_err":  ee_pos_err
    }


if __name__ == "__main__":
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/app/logs/replay_trajectory/{run_id}"
    task_cfg_path = f"IMPACT/trajectory_replay/default.yaml"
    traj_path = "/app/data/droid_1.0.1"
    rendering_mode = "r"
    ep_names = [d for d in os.listdir(traj_path) if os.path.isdir(os.path.join(traj_path, d))]

    set_sim_config(rendering_mode=rendering_mode)
    env = RealmEnvironmentDynamic(
        config_path="/app/realm/config",
        task_cfg_path=task_cfg_path,
        perturbations=["Default"],
        rendering_mode=rendering_mode,
        robot="DROID_no_wrist_cam",
        no_rendering=True
    )

    for traj_id in range(len(ep_names)):
        # TODO: check file structure and fix paths if needed
        traj_qpos_actions = np.load(f"{traj_path}/{ep_names[traj_id]}/action_qpos.npy")
        traj_qpos_gt = np.load(f"{traj_path}/{ep_names[traj_id]}/states_qpos.npy")
        traj_ee_gt = np.load(f"{traj_path}/{ep_names[traj_id]}/states_ee.npy")
        # for i in len(traj_ee_gt):
        #     traj_ee_gt[i, 3:6] = flip_pose_pointing_down(traj_ee_gt[i, 3:6])

        res_dict = replay_traj(traj_qpos_actions, traj_qpos_gt, traj_ee_gt)

        qpos_err = res_dict["qpos_err"]
        ee_pos_err = res_dict["ee_pos_err"]
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot joint errors
        axes[0].plot(qpos_err)
        axes[0].set_title(f"Joint Errors for {ep_names[traj_id]}")
        axes[0].set_ylabel("Error (rad)")
        axes[0].set_xlabel("Time steps")
        axes[0].legend([f"Joint {i}" for i in range(7)], loc='upper right')
        axes[0].grid(True)
        
        # Plot EE xyz errors
        axes[1].plot(ee_pos_err)
        axes[1].set_title(f"End-Effector XYZ Errors for {ep_names[traj_id]}")
        axes[1].set_ylabel("Error (m)")
        axes[1].set_xlabel("Time steps")
        axes[1].legend(['X', 'Y', 'Z'], loc='upper right')
        axes[1].grid(True)
        
        plt.tight_layout()
        
        plots_dir = os.path.join(log_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, f"{ep_names[traj_id]}.png")
        plt.savefig(plot_path)
        plt.close(fig)

    og.shutdown()
    sys.exit(0)
