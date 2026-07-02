import optuna
import sys
import copy
import os
import json
import argparse
import torch
from datetime import datetime
from main import get_args, run
from dataloaders.base import RafDB_perm

DEFAULT_PROPOSED_AGENT = "ProposedFramework"
PROPOSED_FAMILY = {
    "ProposedFramework",
    "ProposedFrameworkNoFairness",
    "ProposedFrameworkMagnitudeRanking",
    "ProposedFrameworkNoFairnessMagnitudeRanking",
}
NO_FAIRNESS_AGENTS = {
    "ProposedFrameworkNoFairness",
    "ProposedFrameworkNoFairnessMagnitudeRanking",
}


def parse_provided_keys(argv):
    keys = set()
    for arg in argv:
        if arg.startswith('--'):
            keys.add(arg[2:].split('=')[0].replace('-', '_'))
    return keys


def validate_proposed_agent(args, provided_keys):
    if args.agent_name in PROPOSED_FAMILY:
        return
    if "agent_name" in provided_keys:
        valid = ", ".join(sorted(PROPOSED_FAMILY))
        raise ValueError(f"Unknown proposed-family agent '{args.agent_name}'. Valid options: {valid}")
    args.agent_name = DEFAULT_PROPOSED_AGENT


def uses_fairness_path(agent_name):
    return agent_name not in NO_FAIRNESS_AGENTS


def objective(trial, base_args, tune_stage, preloaded_datasets=None):
    """
    Optuna objective function for the different tuning stages.
    """
    args = copy.deepcopy(base_args)

    args.save_weights = False

    if tune_stage == "global":
        # Tune Global Hyperparameters
        args.lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        args.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        schedule_val = trial.suggest_int("schedule", 1, 40)
        args.schedule = [schedule_val]
        args.optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam", "RMSprop"])
        args.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    elif tune_stage == "baseline":
        if args.agent_name in ["SI", "EWC"]:
            args.reg_coef = trial.suggest_float("reg_coef", 0.1, 1000.0, log=True)

        elif args.agent_name == "MagnitudePruning":
            args.prune_amount = trial.suggest_float("prune_amount", 0.7, 0.99)
            args.prune_retrain_epochs = trial.suggest_int("prune_retrain_epochs", 0, 5)

        elif args.agent_name == "BiasAwarePruning":
            args.target_acc = trial.suggest_float("target_acc", 0.75, 0.85)
            args.init_prune_rate = trial.suggest_float("init_prune_rate", 0.7, 0.99)
            args.alpha = trial.suggest_float("alpha", 0.01, 0.10)
            args.inc_prune_rate = trial.suggest_float("inc_prune_rate", 0.5, 0.9)
            args.prune_retrain_epochs = trial.suggest_int("prune_retrain_epochs", 0, 5)

        elif args.agent_name == "SIMagnitudePruning":
            args.reg_coef = trial.suggest_float("reg_coef", 0.1, 1000.0, log=True)
            args.prune_amount = trial.suggest_float("prune_amount", 0.7, 0.99)
            args.prune_retrain_epochs = trial.suggest_int("prune_retrain_epochs", 0, 5)

    elif tune_stage in ["proposed_stage1", "proposed_full"]:
        args.target_acc = trial.suggest_float("target_acc", 0.75, 0.85)
        args.reg_coef = trial.suggest_float("reg_coef", 0.1, 1000.0, log=True)
        if uses_fairness_path(args.agent_name):
            args.lambda_bias = trial.suggest_float("lambda_bias", 0.1, 1000.0, log=True)
        args.init_prune_rate = trial.suggest_float("init_prune_rate", 0.7, 0.99)
        args.inc_prune_rate = trial.suggest_float("inc_prune_rate", 0.5, 0.99)
        if uses_fairness_path(args.agent_name):
            args.alpha = trial.suggest_float("alpha", 0.01, 0.10)
        args.prune_retrain_epochs = trial.suggest_int("prune_retrain_epochs", 0, 5)

    elif tune_stage == "proposed_stage2":
        if uses_fairness_path(args.agent_name):
            args.lambda_bias = trial.suggest_float("lambda_bias", 0.1, 1000.0, log=True)
            args.alpha = trial.suggest_float("alpha", 0.01, 0.20)
        args.init_prune_rate = trial.suggest_float("init_prune_rate", 0.5, 0.99)

    acc_table, task_names, f1_table, param_count = run(args, preloaded_datasets)
    if tune_stage != "global":
        print(f"Trial {trial.number}: Parameter Count = {param_count}")
    final_task = 'All' if 'All' in acc_table else task_names[-1]

    if tune_stage == "global":
        f1_score_overall = f1_table['All']['All']
        return f1_score_overall
    else:
        accs = [acc_table[final_task][val_name] for val_name in task_names]
        avg_acc = sum(accs) / len(accs)
        fairness = min(accs) / max(accs) if max(accs) > 0 else 0.0
        return avg_acc, fairness, param_count

def save_config(params, filename):
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)

def load_config(args, filename):
    if not os.path.exists(filename):
        print(f"Warning: Config file {filename} not found.")
        return args

    with open(filename, 'r') as f:
        params = json.load(f)

    for k, v in params.items():
        if k == "schedule":
            if isinstance(v, str):
                setattr(args, k, [int(x) for x in v.split(',')])
            elif isinstance(v, int):
                setattr(args, k, [v])
            else:
                setattr(args, k, v)
        else:
            setattr(args, k, v)
    print(f"Loaded config from {filename}: {params}")
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning Script")
    parser.add_argument('--tune_stage', type=str, required=True, choices=["global", "baseline", "proposed_stage1", "proposed_stage2", "proposed_full"], help="The stage of tuning to perform.")
    parser.add_argument('--n_trials', type=int, default=5, help="Number of Optuna trials.")

    optuna_args, remaining_argv = parser.parse_known_args()
    provided_keys = parse_provided_keys(remaining_argv)

    base_args = get_args(remaining_argv)

    output_dir = "configs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Starting Optuna Hyperparameter Optimization - Stage: {optuna_args.tune_stage}")

    if optuna_args.tune_stage == "global":
        base_args.offline_training = True # Global hyperparameters are tuned using offline joint training
        print("Enforcing offline_training=True for global tuning.")

    elif optuna_args.tune_stage in ["baseline", "proposed_stage1", "proposed_stage2", "proposed_full"]:
        global_config_path = os.path.join(output_dir, "global_config.json")
        base_args = load_config(base_args, global_config_path)

        if optuna_args.tune_stage == "proposed_stage1":
            base_args.category = "gender"
            base_args.train_aug = False
            base_args.agent_type = "customization"
            validate_proposed_agent(base_args, provided_keys)
            print(f"Enforcing gender_noaug and {base_args.agent_name} for Stage 1.")

        elif optuna_args.tune_stage == "proposed_stage2":
            validate_proposed_agent(base_args, provided_keys)
            proposed_config_path = os.path.join(output_dir, f"proposed_base_config_{base_args.agent_name}.json")
            if not os.path.exists(proposed_config_path):
                proposed_config_path = os.path.join(output_dir, "proposed_base_config.json")
            base_args = load_config(base_args, proposed_config_path)

            base_args.agent_type = "customization"
            validate_proposed_agent(base_args, provided_keys)
            print(f"Loaded Stage 1 parameters. Enforcing {base_args.agent_name} for condition: {base_args.category} (Augmented: {base_args.train_aug}).")

        elif optuna_args.tune_stage == "proposed_full":
            base_args.agent_type = "customization"
            validate_proposed_agent(base_args, provided_keys)
            print(f"Tuning all parameters for {base_args.agent_name} on condition: {base_args.category} (Augmented: {base_args.train_aug}).")

    print("Preloading dataset")
    train_dataset_splits, val_dataset_splits, task_output_space = RafDB_perm(base_args.category, base_args.train_aug, base_args.skip_unsure)
    if base_args.debug:
        print("DEBUG MODE: using only a small portion of the dataset")
        for k in train_dataset_splits.keys():
            train_dataset_splits[k] = torch.utils.data.Subset(train_dataset_splits[k], range(min(50, len(train_dataset_splits[k]))))
        for k in val_dataset_splits.keys():
            val_dataset_splits[k] = torch.utils.data.Subset(val_dataset_splits[k], range(min(50, len(val_dataset_splits[k]))))
    preloaded_datasets = (train_dataset_splits, val_dataset_splits, task_output_space)

    if optuna_args.tune_stage == "global":
        study = optuna.create_study(direction="maximize")
    else:
        study = optuna.create_study(directions=["maximize", "maximize", "minimize"])
        
    study.optimize(lambda trial: objective(trial, base_args, optuna_args.tune_stage, preloaded_datasets), n_trials=optuna_args.n_trials)
    
    print("\n" + "="*50)
    print("Optimization Complete!")
    print("="*50)
    
    if optuna_args.tune_stage == "global":
        print("Best Trial:")
        best_trial = study.best_trial
        print(f"  F1 Score: {best_trial.value:.4f}")
        print("  Hyperparameters: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
    else:
        print("Pareto-Optimal Trials:")
        for i, trial in enumerate(study.best_trials):
            print(f"\n  --- Trade-off Option {i+1} ---")
            print(f"  Accuracy: {trial.values[0]:.4f} | Fairness: {trial.values[1]:.4f} | Param Count: {trial.values[2]:.0f}")
            print("  Hyperparameters: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
            
    if optuna_args.tune_stage == "global":
        best_trial = study.best_trial
        save_config(best_trial.params, os.path.join(output_dir, "global_config.json"))
        print(f"Saved global config to {output_dir}/global_config.json")
    elif len(study.best_trials) > 0:
        best_trial = study.best_trials[0] 
        
        res_dir = "results_hyperparam"
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        aug_str = "augmented" if base_args.train_aug else "non_augmented"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if optuna_args.tune_stage == "proposed_stage1":
            config_to_save = {
                "reg_coef": best_trial.params["reg_coef"],
                "inc_prune_rate": best_trial.params["inc_prune_rate"]
            }
            agent_config_path = os.path.join(output_dir, f"proposed_base_config_{base_args.agent_name}.json")
            save_config(config_to_save, agent_config_path)
            print(f"Saved proposed base config to {agent_config_path}")
            if base_args.agent_name == DEFAULT_PROPOSED_AGENT:
                save_config(config_to_save, os.path.join(output_dir, "proposed_base_config.json"))
        elif optuna_args.tune_stage == "proposed_full":
            config_name = f"config_{optuna_args.tune_stage}_{base_args.agent_name}_{base_args.category}_{aug_str}_{timestamp}.json"
            config_path = os.path.join(res_dir, config_name)
            config_to_save = {
                key: best_trial.params[key]
                for key in [
                    "target_acc", "reg_coef", "lambda_bias", "init_prune_rate",
                    "inc_prune_rate", "alpha", "prune_retrain_epochs"
                ]
                if key in best_trial.params
            }
            save_config(config_to_save, config_path)
            print(f"Saved proposed full config to {config_path}")

        out_file = os.path.join(res_dir, f"best_trials_{optuna_args.tune_stage}_{base_args.agent_name}_{base_args.category}_{aug_str}_{timestamp}.json")

        trials_list = []
        
        for i, trial in enumerate(study.best_trials):
            trials_list.append({
                "trade_off_option": i + 1,
                "trial_number": trial.number,
                "accuracy": trial.values[0],
                "fairness": trial.values[1],
                "param_count": trial.values[2],
                "params": trial.params
            })

        with open(out_file, "w") as f:
            json.dump(trials_list, f, indent=4)
        print(f"Detailed trial results saved to {out_file}")
