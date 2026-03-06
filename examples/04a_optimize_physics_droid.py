import argparse
import os
import sys
import numpy as np
import torch
from datetime import datetime
import cma

import omnigibson as og

from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.eval import set_sim_config
from realm.utils import cost_function


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dynamic sim evals")
    parser.add_argument('--max_eps', type=int, required=False, default=5)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/app/logs/replay_trajectory/{run_id}"
    task_cfg_path = f"other/trajectory_replay/default.yaml"
    rendering_mode = "r"
    traj_path = "/app/data/droid_1.0.1/extracted_eps/chunk-000/"

    set_sim_config(rendering_mode=rendering_mode)
    env = RealmEnvironmentDynamic(
        config_path="/app/realm/config",
        task_cfg_path=task_cfg_path,
        perturbations=["Default"],
        rendering_mode=rendering_mode,
        robot="DROID_no_wrist_cam",
        no_rendering=True
    )

    friction = np.array([0.05, 0.15, 0.25, 0.15, 0.75, 0.15, 0.50])
    armature = np.array([0.50, 0.20, 0.50, 0.20, 0.25, 0.00, 0.25])
    initial_flat_params = np.concatenate((friction, armature))

    # sigma0 is the initial step-size (standard deviation) for the search.
    # A good starting point is often 0.1 to 1.0, depending on the scale of your variables.
    initial_sigma = 0.5 #0.075

    # --- 4. Define CMA-ES options (optional but recommended) ---
    # See https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html#cma.evolution_strategy.CMAOptions
    # for a full list of options.
    options = {
        'seed': 42,  # For reproducibility
        'maxfevals': 55, #500,  # Maximum number of function evaluations
        'popsize': 11, #100,  # Population size (number of solutions evaluated per iteration)
        'tolfun': 1.0,  # Tolerance for cost function value
        'tolx': 1e-3,  # Tolerance for variable changes
        'verb_disp': 1,  # Display progress every X iterations
        'verbose': -9,  # Reduce verbosity of intermediate output
        'bounds': [0, 2]  # Example for boundary constraints on *each* flattened parameter
    }

    # --- 5. Run the optimization ---
    from realm.utils import set_flat_physics_params
    
    def replay_error(x):
        set_flat_physics_params(env, x)
        return cost_function(env=env, traj_path=traj_path, max_eps=args.max_eps)

    os.makedirs(log_dir, exist_ok=True)

    es = cma.CMAEvolutionStrategy(initial_flat_params, initial_sigma, inopts=options)
    
    while not es.stop():
        solutions = es.ask()
        costs = [replay_error(x) for x in solutions]
        es.tell(solutions, costs)
        es.disp()
        
        # Periodic logging of the best parameters
        np.save(f"{log_dir}/best_params.npy", es.result.xbest)

    # --- 6. Access the results ---
    best_flat_params = es.result.xbest  # Best solution found
    best_cost = es.result.fbest  # Cost value of the best solution
    num_evaluations = es.result.evals  # Number of function evaluations
    num_iterations = es.result.iterations  # Number of iterations
    mean_params_flat = es.result.xmean  # Mean of the population

    print(options)

    print("\n--- Optimization Results ---")
    print(f"Best cost found: {best_cost}")
    print(f"Number of function evaluations: {num_evaluations}")
    print(f"Number of iterations: {num_iterations}")

    # Unflatten the best parameters to get them back in their original array forms
    optimal_friction = best_flat_params[:7]
    optimal_armature = best_flat_params[7:14]

    print("\nOptimized friction:\n", optimal_friction)
    print("\nOptimized armature:\n", optimal_armature)

    # Save results if needed
    os.makedirs(log_dir, exist_ok=True)
    np.save(f"{log_dir}/best_params.npy", best_flat_params)

    og.shutdown()
    sys.exit(0)
