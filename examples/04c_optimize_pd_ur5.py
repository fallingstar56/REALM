import argparse
import sys
import os
import numpy as np
import torch
from datetime import datetime
import cma

import omnigibson as og

from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.eval import set_sim_config
from realm.utils import cost_function, set_flat_physics_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UR5 dynamic sim physics optimization")
    parser.add_argument('--max_eps', type=int, required=False, default=10)
    parser.add_argument('--seed', type=int, required=False, default=42)
    parser.add_argument('--initial_sigma', type=float, required=False, default=0.5)
    parser.add_argument('--popsize', type=int, required=False, default=10)
    parser.add_argument('--max_evals', type=int, required=False, default=55)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/app/logs/optimize_physics_ur5/{run_id}"
    task_cfg_path = f"other/trajectory_replay/default.yaml"
    rendering_mode = "r"
    traj_path = "/app/data/RoboMIND2.0-UR5/data/ur/"
    robot = "UR5"
    dof = 6

    set_sim_config(rendering_mode=rendering_mode, robot=robot)
    env = RealmEnvironmentDynamic(
        config_path="/app/realm/config",
        task_cfg_path=task_cfg_path,
        perturbations=["Default"],
        rendering_mode=rendering_mode,
        robot=robot,
        no_rendering=True
    )

    initial_flat_params = np.array([300, 0])

    initial_sigma = args.initial_sigma

    options = {
        'seed': args.seed,
        'maxfevals': args.max_evals,
        'popsize': args.popsize,
        'tolfun': 1e-3,
        'tolx': 1e-4,
        'verb_disp': 1,
        'verbose': -9,
        'bounds': [0, 1000]
    }

    # --- Run the optimization ---
    def objective(x):
        controller_config = {
            "arm_0": {
                "name": "IndividualJointPDController",
                "motor_type": "position",
                "control_freq": 30,
                "use_delta_commands": False,
                "use_impedances": True,
                "use_gravity_compensation": False,
                "use_cc_compensation": False,
                "command_output_limits": None,
                "command_input_limits": None,
                "kp": x[0],
                "kd": x[1],
                "max_effort": [ 150.0, 150.0, 150.0, 28.0, 28.0, 28.0 ],
                "min_effort": [ -150.0, -150.0, -150.0, -28.0, -28.0, -28.0 ]
            }
        }
        env.robot.reload_controllers(controller_config=controller_config)

        # Evaluate cost using replayed trajectories
        return cost_function(env=env, traj_path=traj_path, max_eps=args.max_eps, dof=dof)

    os.makedirs(log_dir, exist_ok=True)

    print(f"Starting CMA-ES optimization for {robot} (DOF={dof})...")
    es = cma.CMAEvolutionStrategy(initial_flat_params, initial_sigma, inopts=options)

    try:
        log_file_path = os.path.join(log_dir, "optimization_log.txt")
        with open(log_file_path, "w") as f:
            f.write("Iteration,Best_Cost,Best_Kp,Best_Kd\n")
            
        iteration = 0
        while not es.stop():
            xs = es.ask()
            costs = [objective(x) for x in xs]
            es.tell(xs, costs)
            
            best_x = es.result.xbest
            best_f = es.result.fbest
            np.save(os.path.join(log_dir, "best_params.npy"), best_x)
            
            with open(log_file_path, "a") as f:
                f.write(f"{iteration},{best_f},{best_x[0]},{best_x[1]}\n")
                
            iteration += 1
    finally:
        final_p = es.result.xbest
        path = os.path.join(log_dir, "final_best_params.npy")
        np.save(path, final_p)
        os.sync()  # Force write to disk
        print(f"Final Best: {final_p} Cost: {es.result.fbest}")
        og.shutdown()
        #sys.exit(0)

    # while not es.stop():
    #     solutions = es.ask()
    #     costs = [objective(x) for x in solutions]
    #     es.tell(solutions, costs)
    #     es.disp()
    #
    #     # Periodic logging of the best parameters
    #     np.save(f"{log_dir}/best_params.npy", es.result.xbest)
    #
    # # --- Results ---
    # best_flat_params = es.result.xbest
    # best_cost = es.result.fbest
    #
    # print("\n--- Optimization Results ---")
    # print(f"Best cost found: {best_cost}")
    # print(f"Optimized friction: {best_flat_params[:dof]}")
    # print(f"Optimized armature: {best_flat_params[dof:]}")
    #
    # # Save results if needed
    # os.makedirs(log_dir, exist_ok=True)
    # np.save(f"{log_dir}/best_params.npy", best_flat_params)
    #
    # og.shutdown()
    # sys.exit(0)
