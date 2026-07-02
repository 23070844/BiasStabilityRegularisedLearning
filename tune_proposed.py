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


def uses_fairness_path(agent_name):
    return agent_name not in NO_FAIRNESS_AGENTS


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


def objective(trial, base_args, preloaded_datasets):
    """
    Optuna objective function to tune:
      - init_prune_rate: 0.90 to 0.99 (linear)
      - reg_coef: 0.01 to 1000.0 (log-uniform)
      - lambda_bias: -100.0 to 100.0 (linear; fairness-path agents only)
      - inc_prune_rate: 0.1 to 0.9 (linear)
      - prune_retrain_epochs: 0 to 5 (integer)

    Objectives:
      1. Maximize accuracy (avg_acc)
      2. Maximize fairness (fairness)
      3. Minimize parameter count (param_count)
    """
    args = copy.deepcopy(base_args)
    args.save_weights = False

    args.init_prune_rate = trial.suggest_float("init_prune_rate", 0.90, 0.99)
    args.reg_coef = trial.suggest_float("reg_coef", 0.01, 1000.0, log=True)
    if uses_fairness_path(args.agent_name):
        args.lambda_bias = trial.suggest_float("lambda_bias", -100.0, 100.0)
    args.inc_prune_rate = trial.suggest_float("inc_prune_rate", 0.1, 0.9)
    args.prune_retrain_epochs = trial.suggest_int("prune_retrain_epochs", 0, 5)

    acc_table, task_names, f1_table, param_count = run(args, preloaded_datasets)
    print(f"Trial {trial.number}: Parameter Count = {param_count}")

    final_task = task_names[-1]
    group_accs = [acc_table[final_task][group] for group in task_names]

    avg_acc = sum(group_accs) / len(group_accs)
    fairness = min(group_accs) / max(group_accs) if max(group_accs) > 0 else 0.0

    print(f"Trial {trial.number}: Avg Acc = {avg_acc:.4f}, Fairness = {fairness:.4f}")

    return avg_acc, fairness, param_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Tuning for the ProposedFramework family")
    parser.add_argument('--n_trials', type=int, default=10, help="Number of Optuna trials.")
    
    script_args, remaining_argv = parser.parse_known_args()
    provided_keys = parse_provided_keys(remaining_argv)

    base_args = get_args(remaining_argv)

    defaults = {
        'gpuid': [0],
        'repeat': 1,
        'model_type': 'resnet',
        'model_name': 'TVResNet18_pretrained_freeze',
        'force_out_dim': 7,
        'no_class_remap': True,
        'optimizer': 'Adam',
        'batch_size': 64,
        'lr': 0.0013701119904548946,
        'weight_decay': 4.3364101535633925e-05,
        'schedule': [18],
        'agent_type': 'customization',
        'agent_name': 'ProposedFramework',
        'target_acc': 0.75,
        'inc_prune_rate': 0.5,
        'alpha': 0.05,
        'prune_retrain_epochs': 5,
        'category': 'gender',
        'skip_unsure': True,
        'train_aug': False
    }

    for key, val in defaults.items():
        if key not in provided_keys:
            setattr(base_args, key, val)

    validate_proposed_agent(base_args, provided_keys)

    print(f"{'='*60}")
    print(f"  Optuna Hyperparameter Tuning for {base_args.agent_name}")
    print(f"  Trials: {script_args.n_trials}")
    print(f"  Category: {base_args.category} | Aug: {base_args.train_aug}")
    print(f"  Model: {base_args.model_type}/{base_args.model_name}")
    print(f"{'='*60}")

    print("\nPreloading dataset")
    train_dataset_splits, val_dataset_splits, task_output_space = RafDB_perm(
        base_args.category, base_args.train_aug, base_args.skip_unsure
    )
    
    if base_args.debug:
        print("DEBUG MODE: using only a small portion of the dataset")
        for k in train_dataset_splits.keys():
            train_dataset_splits[k] = torch.utils.data.Subset(
                train_dataset_splits[k], range(min(50, len(train_dataset_splits[k])))
            )
        for k in val_dataset_splits.keys():
            val_dataset_splits[k] = torch.utils.data.Subset(
                val_dataset_splits[k], range(min(50, len(val_dataset_splits[k])))
            )
            
    preloaded_datasets = (train_dataset_splits, val_dataset_splits, task_output_space)

    study = optuna.create_study(directions=["maximize", "maximize", "minimize"])

    study.optimize(
        lambda trial: objective(trial, base_args, preloaded_datasets),
        n_trials=script_args.n_trials
    )

    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)

    print("\nPareto-Optimal Trials:")
    for i, trial in enumerate(study.best_trials):
        print(f"\n  --- Trade-off Option {i+1} ---")
        print(f"  Accuracy: {trial.values[0]:.4f} | Fairness: {trial.values[1]:.4f} | Param Count: {trial.values[2]:.0f}")
        print("  Hyperparameters: ")
        print(f"    init_prune_rate: {trial.params.get('init_prune_rate'):.4f}")
        print(f"    reg_coef: {trial.params.get('reg_coef'):.6f}")
        if "lambda_bias" in trial.params:
            print(f"    lambda_bias: {trial.params.get('lambda_bias'):.6f}")
        print(f"    inc_prune_rate: {trial.params.get('inc_prune_rate'):.4f}")
        print(f"    prune_retrain_epochs: {trial.params.get('prune_retrain_epochs')}")

    res_dir = "results_hyperparam"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    aug_str = "augmented" if base_args.train_aug else "non_augmented"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    trials_list = []
    for i, trial in enumerate(study.best_trials):
        entry = {
            "trade_off_option": i + 1,
            "trial_number": trial.number,
            "accuracy": trial.values[0],
            "fairness": trial.values[1],
            "param_count": int(trial.values[2]),
            "params": {
                key: trial.params[key]
                for key in [
                    "init_prune_rate", "reg_coef", "lambda_bias",
                    "inc_prune_rate", "prune_retrain_epochs"
                ]
                if key in trial.params
            }
        }
        trials_list.append(entry)

    out_file = os.path.join(
        res_dir,
        f"best_trials_{base_args.agent_name}_{base_args.category}_{aug_str}_{timestamp}.json"
    )
    with open(out_file, "w") as f:
        json.dump(trials_list, f, indent=4)
    print(f"\nDetailed trial results saved to {out_file}")
