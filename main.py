import os
import sys
import argparse
import json
import torch
import numpy as np
from random import shuffle
from collections import OrderedDict
import dataloaders.base
from dataloaders.datasetGen import SplitGen, PermutedGen, RafdbTask
import agents
from dataloaders.base import RafDB_perm
from datetime import datetime

PROPOSED_FAMILY_AGENTS = {
    'ProposedFramework',
    'ProposedFrameworkNoFairness',
    'ProposedFrameworkMagnitudeRanking',
    'ProposedFrameworkNoFairnessMagnitudeRanking',
}

def save_agent_weights(agent, args, suffix):
    if not os.path.exists('weights'):
        os.makedirs('weights')
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cat_name = args.category if args.train_aug == False else args.category + "_augmented"
    weight_filename = "weights/" + args.model_type + "_" + args.agent_name + "_" + str(args.reg_coef) + "_" + cat_name + "_epoch" + str(args.schedule[-1]) + "_" + suffix + "_" + date_str
    agent.save_model(weight_filename)

def run(args, preloaded_datasets=None):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    print("category: ",args.category)
    
    if preloaded_datasets is not None:
        train_dataset_splits, val_dataset_splits, task_output_space = preloaded_datasets
    else:
        train_dataset_splits, val_dataset_splits, task_output_space = RafDB_perm(args.category, args.train_aug, args.skip_unsure)
        
        if args.debug:
            print("DEBUG MODE: using only a small portion of the dataset")
            for k in train_dataset_splits.keys():
                train_dataset_splits[k] = torch.utils.data.Subset(train_dataset_splits[k], range(min(50, len(train_dataset_splits[k]))))
            for k in val_dataset_splits.keys():
                val_dataset_splits[k] = torch.utils.data.Subset(val_dataset_splits[k], range(min(50, len(val_dataset_splits[k]))))
            
    agent_config = {'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay,'schedule': args.schedule,
                    'model_type':args.model_type, 'model_name': args.model_name, 'model_weights':args.model_weights,
                    'out_dim':{'All':args.force_out_dim} if args.force_out_dim>0 else task_output_space,
                    'optimizer':args.optimizer,
                    'print_freq':args.print_freq, 'gpuid': args.gpuid,
                    'reg_coef':args.reg_coef, 'prune_amount':args.prune_amount,
                    'target_acc':args.target_acc, 'alpha':args.alpha,
                    'init_prune_rate':args.init_prune_rate, 'inc_prune_rate':args.inc_prune_rate,
                    'prune_retrain_epochs':args.prune_retrain_epochs, 'lambda_bias':args.lambda_bias}
    agent_type = args.agent_type
    agent_name = args.agent_name
    if agent_type not in agents.__dict__ or agent_name not in agents.__dict__[agent_type].__dict__:
        found = False
        for module_name in ['default', 'regularization', 'customization', 'exp_replay']:
            if module_name in agents.__dict__ and agent_name in agents.__dict__[module_name].__dict__:
                agent_type = module_name
                found = True
                break
        if not found:
            raise KeyError(f"Agent class '{agent_name}' not found in any of the agent modules under agents.")
            
    agent = agents.__dict__[agent_type].__dict__[agent_name](agent_config)
    print('# of parameters:',agent.count_parameter())

    task_names = sorted(list(task_output_space.keys()), key=int)
    print('Task order:',task_names)
    if args.rand_split_order:
        shuffle(task_names)
        print('Shuffled task order:', task_names)


    acc_table = OrderedDict()
    f1_table = OrderedDict()
    if args.offline_training:  # Non-incremental learning / offline_training / measure the upper-bound performance
        train_dataset_all = torch.utils.data.ConcatDataset(train_dataset_splits.values())
        val_dataset_all = torch.utils.data.ConcatDataset(val_dataset_splits.values())
        train_loader = torch.utils.data.DataLoader(train_dataset_all,
                                                   batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset_all,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        agent.learn_batch(train_loader, val_loader)

        if args.save_weights:
            save_agent_weights(agent, args, "offline")

        acc_table['All'] = {}
        f1_table["All"] = {}
        val_loader_all = torch.utils.data.DataLoader(val_dataset_all, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        acc_table['All']['All'], f1_table['All']['All'] = agent.validation(val_loader_all)
        for j in range(len(task_names)):
            val_name = task_names[j]
            print('validation split name:', val_name)
            val_data = val_dataset_splits[val_name] 
            val_loader = torch.utils.data.DataLoader(val_data,
                                                     batch_size=args.batch_size, shuffle=False,
                                                     num_workers=args.workers)
            acc_table["All"][val_name], f1_table["All"][val_name] = agent.validation(val_loader)

    else:  # Incremental learning
        # Create a global validation loader for bias evaluation across all demographic groups
        val_dataset_all = torch.utils.data.ConcatDataset(val_dataset_splits.values())
        global_val_loader = torch.utils.data.DataLoader(val_dataset_all,
                                                  batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        for i in range(len(task_names)):
            print(len(train_dataset_splits[task_names[i]]))
        for i in range(len(task_names)):
            train_name = task_names[i]
            print('======================',train_name,'=======================')
            train_loader = torch.utils.data.DataLoader(train_dataset_splits[train_name],
                                                        batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
            val_loader = torch.utils.data.DataLoader(val_dataset_splits[train_name],
                                                      batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

            if args.incremental_class:
                agent.add_valid_output_dim(task_output_space[train_name])

            # Learn
            kwargs = {}
            if args.agent_name in ['BiasAwarePruning'] or args.agent_name in PROPOSED_FAMILY_AGENTS:
                kwargs['global_val_loader'] = global_val_loader
            if args.agent_name in ['SIMagnitudePruning'] or args.agent_name in PROPOSED_FAMILY_AGENTS:
                kwargs['is_last_task'] = (i == len(task_names) - 1)
                
            agent.learn_batch(train_loader, val_loader, **kwargs)

            if args.save_weights and args.save_weight_mode == 'each_task':
                save_agent_weights(agent, args, "task" + str(i))

            # Evaluate
            acc_table[train_name] = OrderedDict()
            f1_table[train_name] = OrderedDict()
            for j in range(len(task_names)):
                val_name = task_names[j]
                print('validation split name:', val_name)
                val_data = val_dataset_splits[val_name] if not args.eval_on_train_set else train_dataset_splits[val_name]
                val_loader = torch.utils.data.DataLoader(val_data,
                                                         batch_size=args.batch_size, shuffle=False,
                                                         num_workers=args.workers)
                acc_table[train_name][val_name], f1_table[train_name][val_name] = agent.validation(val_loader)

        if args.save_weights and args.save_weight_mode == 'end':
            save_agent_weights(agent, args, "end")

    return acc_table, task_names, f1_table, agent.count_parameter()

def get_args(argv):
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--model_type', type=str, default='mlp', help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_name', type=str, default='MLP', help="The name of actual model for the backbone")
    parser.add_argument('--force_out_dim', type=int, default=2, help="Set 0 to let the task decide the required output dimension")
    parser.add_argument('--agent_type', type=str, default='default', help="The type (filename) of agent")
    parser.add_argument('--agent_name', type=str, default='NormalNN', help="The class name of agent")
    parser.add_argument('--optimizer', type=str, default='SGD', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='MNIST', help="MNIST(default)|CIFAR10|CIFAR100")
    parser.add_argument('--n_permutation', type=int, default=0, help="Enable permuted tests when >0")
    parser.add_argument('--first_split_size', type=int, default=2)
    parser.add_argument('--other_split_size', type=int, default=2)
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
                        help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,6 ...] -> [0,1,2 ...]")
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                        help="Randomize the order of splits")
    parser.add_argument('--workers', type=int, default=3, help="#Thread for dataloader")
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--schedule', nargs="+", type=int, default=[2],
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")
    parser.add_argument('--print_freq', type=float, default=100, help="Print the log at every x iteration")
    parser.add_argument('--model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--reg_coef', nargs="+", type=float, default=[0.], help="The coefficient for regularization. Larger means less plasilicity. Give a list for hyperparameter search.")
    parser.add_argument('--eval_on_train_set', dest='eval_on_train_set', default=False, action='store_true',
                        help="Force the evaluation on train set")
    parser.add_argument('--offline_training', dest='offline_training', default=False, action='store_true',
                        help="Non-incremental learning by make all data available in one batch. For measuring the upperbound performance.")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--incremental_class', dest='incremental_class', default=False, action='store_true',
                        help="The number of output node in the single-headed model increases along with new categories.")
    parser.add_argument('--category', type=str, default='gender', help="The Category (gender, race, age)") 
    parser.add_argument('--save_weights', dest='save_weights', default=False, action='store_true',
                        help="Save weights after each training")
    parser.add_argument('--save_weight_mode', '--save_weights_mode', dest='save_weight_mode',
                        type=str, default='each_task', choices=['each_task', 'end'],
                        help="For incremental training with --save_weights, save after each task or only once at the end. Offline training always saves once.")
    parser.add_argument('--skip_unsure', type=bool, default=False, help="Exclude unsure gender")
    parser.add_argument('--prune_amount', type=float, default=0.0, help="The proportion of weights to prune using magnitude pruning (0.0 means no pruning).")
    parser.add_argument('--target_acc', type=float, default=0.85, help="Target Accuracy A for Bias-Aware Pruning (absolute value).")
    parser.add_argument('--alpha', type=float, default=0.05, help="Significance level for chi2 test.")
    parser.add_argument('--init_prune_rate', type=float, default=0.95, help="Initial edge prune rate E for Coarse Pruning.")
    parser.add_argument('--inc_prune_rate', type=float, default=0.50, help="Incremental edge prune rate beta for Fine Pruning.")
    parser.add_argument('--prune_retrain_epochs', type=int, default=1, help="Number of epochs to retrain during pruning loops.")
    parser.add_argument('--lambda_bias', type=float, default=1.0, help="Lambda Bias (Fairness penalty weight) for Proposed Framework.")
    parser.add_argument('--debug', dest='debug', default=False, action='store_true', help="Run the code using only a very small portion of the data for debugging purposes")
    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    reg_coef_list = args.reg_coef
    avg_final_acc = {}

    print("Preloading dataset for repeats...")
    train_dataset_splits, val_dataset_splits, task_output_space = RafDB_perm(args.category, args.train_aug, args.skip_unsure)
    if args.debug:
        print("DEBUG MODE: using only a small portion of the dataset")
        for k in train_dataset_splits.keys():
            train_dataset_splits[k] = torch.utils.data.Subset(train_dataset_splits[k], range(min(50, len(train_dataset_splits[k]))))
        for k in val_dataset_splits.keys():
            val_dataset_splits[k] = torch.utils.data.Subset(val_dataset_splits[k], range(min(50, len(val_dataset_splits[k]))))
    preloaded_datasets = (train_dataset_splits, val_dataset_splits, task_output_space)

    for reg_coef in reg_coef_list:
        args.reg_coef = reg_coef
        avg_final_acc[reg_coef] = np.zeros(args.repeat)
        for r in range(args.repeat):
            acc_table, task_names, f1_table, param_count = run(args, preloaded_datasets)
            print(acc_table)
            
            name = args.category if args.train_aug == False else args.category + "_augmented"
            if not os.path.exists(f'results/{args.category}'):
                os.makedirs(f'results/{args.category}')
            with open('results/' + args.category + '/'  + name + '.txt', 'a') as f:
                f.write("\n" + str(acc_table) + "f1: " +  str(f1_table) + " repeat:" + str(r) + " reg_coef: " + str(reg_coef)  + " " +  str(args.agent_name))

            acc_json = []
            for train_task, val_dict in acc_table.items():
                val_list = [{"task": str(val_task), "value": float(val)} for val_task, val in val_dict.items()]
                acc_json.append({
                    "task": str(train_task),
                    "validation": val_list
                })

            f1_json = []
            for train_task, val_dict in f1_table.items():
                val_list = [{"task": str(val_task), "value": float(val)} for val_task, val in val_dict.items()]
                f1_json.append({
                    "task": str(train_task),
                    "validation": val_list
                })

            result_dict = {
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M"),
                "params": vars(args),
                "results": {
                    "acc": acc_json,
                    "f1": f1_json
                },
                "param_count": param_count
            }
            
            json_dir = 'results_json'
            if not os.path.exists(json_dir):
                os.makedirs(json_dir)
                
            json_filename = os.path.join(json_dir, 'all_results.jsonl')
            
            with open(json_filename, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_dict) + '\n')

            if args.offline_training: 
                avg_acc_history = [0]
                cls_acc_sum = 0

                for j in range(len(task_names)):
                    val_name = task_names[j]
                    cls_acc_sum += acc_table['All'][val_name]

                avg_acc_history[0] = cls_acc_sum / 2
                print('Task', 'All', 'average acc:', avg_acc_history[0])
            else: 
                avg_acc_history = [0] * len(task_names)
                for i in range(len(task_names)):
                    train_name = task_names[i]
                    cls_acc_sum = 0
                    for j in range(i + 1):
                        val_name = task_names[j]
                        cls_acc_sum += acc_table[train_name][val_name]
                    avg_acc_history[i] = cls_acc_sum / (i + 1)
                    print('Task', train_name, 'average acc:', avg_acc_history[i])

            avg_final_acc[reg_coef][r] = avg_acc_history[-1]

            print('===Summary of experiment repeats:',r+1,'/',args.repeat,'===')
            print('The regularization coefficient:', args.reg_coef)
            print('The last avg acc of all repeats:', avg_final_acc[reg_coef])
            print('mean:', avg_final_acc[reg_coef].mean(), 'std:', avg_final_acc[reg_coef].std())
    for reg_coef,v in avg_final_acc.items():
        print('reg_coef:', reg_coef,'mean:', avg_final_acc[reg_coef].mean(), 'std:', avg_final_acc[reg_coef].std())
