import os
import json
import numpy as np
import argparse
from datetime import datetime

def BWT2(matrix):
    """
    Calculates the standard Backward Transfer (BWT) score.
    """
    task_num = matrix.shape[0]
    bwt_scores = [] 
    
    if task_num <= 1 or matrix.shape[0] != matrix.shape[1]:
        return 0.0
        
    final_task_row = matrix[task_num - 1]
    for j in range(task_num - 1):
        forgetting = final_task_row[j] - matrix[j, j]
        bwt_scores.append(forgetting)
    return np.mean(bwt_scores)

def BWT(matrix):
    task_num = matrix.shape[0]
    sum1 = 0
    if task_num <= 1 or matrix.shape[0] != matrix.shape[1]: 
        return 0.0
        
    for i in range(1,task_num):
        for j in range(0,i):
            sum1 += matrix[i,j] - matrix[j,j]
            
    sum1 = sum1 / (task_num * (task_num-1) / 2)
    return sum1

def load_json_to_matrix(data):
    acc_data = data['results'].get('acc', [])
    f1_data = data['results'].get('f1', [])
    
    train_task_names = []
    val_task_names = []
    
    for item in acc_data:
        if item['task'] not in train_task_names:
            train_task_names.append(item['task'])
        for val_item in item['validation']:
            if val_item['task'] == 'All':
                continue
            if val_item['task'] not in val_task_names:
                val_task_names.append(val_item['task'])
            
    num_train = len(train_task_names)
    num_val = len(val_task_names)
    
    if num_train == 0 or num_val == 0:
        return np.zeros((1,1)), np.zeros((1,1)), data.get('params', {}), data.get('param_count', 'N/A')
        
    train_to_idx = {name: idx for idx, name in enumerate(train_task_names)}
    val_to_idx = {name: idx for idx, name in enumerate(val_task_names)}
    
    acc_matrix = np.zeros((num_train, num_val))
    f1_matrix = np.zeros((num_train, num_val))
    
    for item in acc_data:
        train_idx = train_to_idx[item['task']]
        for val_item in item['validation']:
            if val_item['task'] in val_to_idx:
                val_idx = val_to_idx[val_item['task']]
                acc_matrix[train_idx, val_idx] = val_item['value']
                
    for item in f1_data:
        train_idx = train_to_idx[item['task']]
        for val_item in item['validation']:
            if val_item['task'] in val_to_idx:
                val_idx = val_to_idx[val_item['task']]
                f1_matrix[train_idx, val_idx] = val_item['value']
                
    return acc_matrix, f1_matrix, data.get('params', {}), data.get('param_count', 'N/A')

def export_to_csv(stats):
    csv_lines = []
    for cat in sorted(stats.keys()):
        # Title
        csv_lines.append(f"{cat.capitalize()}")
        
        # BWT 1
        csv_lines.append(",BWT scores")
        csv_lines.append(",Method,BWT scores")
        csv_lines.append(",,Without Augmentation,With Augmentation")
        
        methods = sorted(stats[cat].keys())
        for method in methods:
            bwt_false = stats[cat][method][False]['bwt'] if stats[cat][method][False] else ""
            bwt_true = stats[cat][method][True]['bwt'] if stats[cat][method][True] else ""
            
            if isinstance(bwt_false, float): bwt_false = f"{bwt_false:.9f}"
            if isinstance(bwt_true, float): bwt_true = f"{bwt_true:.9f}"
            
            csv_lines.append(f",{method},{bwt_false},{bwt_true}")
            
        csv_lines.append("")
        
        # BWT 2
        csv_lines.append(",BWT2 scores")
        csv_lines.append(",Method,BWT2 scores")
        csv_lines.append(",,Without Augmentation,With Augmentation")
        
        for method in methods:
            bwt2_false = stats[cat][method][False]['bwt2'] if stats[cat][method][False] else ""
            bwt2_true = stats[cat][method][True]['bwt2'] if stats[cat][method][True] else ""
            
            if isinstance(bwt2_false, float): bwt2_false = f"{bwt2_false:.9f}"
            if isinstance(bwt2_true, float): bwt2_true = f"{bwt2_true:.9f}"
            
            csv_lines.append(f",{method},{bwt2_false},{bwt2_true}")
            
        csv_lines.append("")
        
        # Param Count
        csv_lines.append(",Parameter Count")
        csv_lines.append(",Method,Parameter Count")
        csv_lines.append(",,Without Augmentation,With Augmentation")
        
        for method in methods:
            pc_false = stats[cat][method][False]['param_count'] if stats[cat][method][False] else ""
            pc_true = stats[cat][method][True]['param_count'] if stats[cat][method][True] else ""
            
            csv_lines.append(f",{method},{pc_false},{pc_true}")
            
        csv_lines.append("")
        csv_lines.append(",Accuracy")
        
        num_tasks = 0
        for method in methods:
            for aug in [False, True]:
                if stats[cat][method][aug]:
                    num_tasks = max(num_tasks, stats[cat][method][aug]['num_tasks'])
        
        if num_tasks > 0:
            empty_cols = "," * num_tasks
            csv_lines.append(f",Method,Without Augmentation{empty_cols},With Augmentation")
            header_tasks = ",".join([f"{cat} {i+1} acc" for i in range(num_tasks)])
            acc_header = f",,{header_tasks},fairness,{header_tasks},fairness"
            csv_lines.append(acc_header)
            
            for method in methods:
                line = [f",{method}"]
                for aug in [False, True]:
                    data = stats[cat][method][aug]
                    if data:
                        accs = data['accs']
                        while len(accs) < num_tasks:
                            accs.append("")
                        accs_str = ",".join(accs)
                        fairness = f"{data['fairness']:.3f}"
                        line.append(f"{accs_str},{fairness}")
                    else:
                        empty_accs = ",".join([""] * num_tasks)
                        line.append(f"{empty_accs},")
                csv_lines.append(",".join(line))
                
        csv_lines.append("")
        
    csv_content = "\n".join(csv_lines)
    
    summary_dir = "results_summary"
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(summary_dir, f"summary_{timestamp}.csv")
    
    with open(csv_path, 'w', encoding='utf-8-sig') as f:
        f.write(csv_content)
        
    print(f"\nCSV summary exported to {csv_path}")

def evaluate_file(filepath):
    print(f"Evaluating JSON results in file: {filepath}")
    
    results_by_config = {}
    grouped_results = {}
    
    try:
        all_results = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    all_results.append(json.loads(line))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return
        
    for idx, item in enumerate(all_results):
        try:
            acc_mat, f1_mat, params, param_count = load_json_to_matrix(item)
            
            agent = params.get('agent_name', 'Unknown')
            reg = params.get('reg_coef', 'Unknown')
            aug = params.get('train_aug', False)
            cat = params.get('category', 'Unknown')
            offline = params.get('offline_training', False)
            config_key = f"Cat:{cat}_{agent}_reg{reg}_aug{aug}_offline{offline}"
            
            if config_key not in results_by_config:
                results_by_config[config_key] = {'acc_mats': [], 'param_counts': []}
                
            results_by_config[config_key]['acc_mats'].append(acc_mat)
            if param_count != 'N/A':
                results_by_config[config_key]['param_counts'].append(param_count)
            
            if cat not in grouped_results:
                grouped_results[cat] = {}
            method = f"{agent}_offline_{reg}" if offline else f"{agent}_{reg}"
            if method not in grouped_results[cat]:
                grouped_results[cat][method] = {False: {'mats': [], 'param_counts': []}, True: {'mats': [], 'param_counts': []}}
            grouped_results[cat][method][aug]['mats'].append(acc_mat)
            if param_count != 'N/A':
                grouped_results[cat][method][aug]['param_counts'].append(param_count)
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            
    if not results_by_config:
        print("No valid results found in the JSON file.")
        return

    print("\n--- Summary of Results ---")
    for config_key, data_dict in results_by_config.items():
        acc_matrices = data_dict['acc_mats']
        param_counts = data_dict['param_counts']
        stacked_acc = np.stack(acc_matrices)
        mean_acc = stacked_acc.mean(axis=0)
        std_acc = stacked_acc.std(axis=0)
        
        bwt1 = BWT(mean_acc)
        bwt2 = BWT2(mean_acc)
        
        print(f"\nConfiguration: {config_key}")
        print(f"Number of runs: {len(acc_matrices)}")
        print(f"BWT: {bwt1:.4f} | BWT2: {bwt2:.4f}")
        print("Mean Accuracy Matrix:")
        print(np.round(mean_acc, 3))
        
        if len(mean_acc) > 0 and len(mean_acc[-1]) > 0:
            final_mean = np.mean(mean_acc[-1])
            print(f"Final Average Accuracy (across all {len(mean_acc[-1])} tasks):", final_mean)
            
    stats = {}
    for cat in grouped_results:
        stats[cat] = {}
        for method in grouped_results[cat]:
            stats[cat][method] = {}
            for aug in [False, True]:
                group_data = grouped_results[cat][method][aug]
                mats = group_data['mats'] if group_data else []
                if len(mats) == 0:
                    stats[cat][method][aug] = None
                    continue
                
                stacked_acc = np.stack(mats)
                mean_acc = stacked_acc.mean(axis=0)
                std_acc = stacked_acc.std(axis=0)
                
                bwt = BWT(mean_acc)
                bwt2 = BWT2(mean_acc)
                
                if len(mean_acc) > 0:
                    last_row_mean = mean_acc[-1]
                    last_row_std = std_acc[-1]
                else:
                    last_row_mean = []
                    last_row_std = []
                
                accs_str = []
                for m, s in zip(last_row_mean, last_row_std):
                    accs_str.append(f"{m:.3f}±{s:.3f}")
                    
                fairness = np.min(last_row_mean) / np.max(last_row_mean) if len(last_row_mean) > 0 and np.max(last_row_mean) > 0 else 0.0
                
                param_counts = group_data['param_counts']
                if param_counts:
                    mean_p = np.mean(param_counts)
                    std_p = np.std(param_counts)
                    param_str = f"{mean_p:.0f}±{std_p:.0f}"
                else:
                    param_str = "N/A"
                    
                stats[cat][method][aug] = {
                    'bwt': bwt,
                    'bwt2': bwt2,
                    'accs': accs_str,
                    'fairness': fairness,
                    'num_tasks': len(last_row_mean),
                    'param_count': param_str
                }
                
    export_to_csv(stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAF-DB JSON results")
    parser.add_argument("--file", type=str, default="results_json/all_results.jsonl", help="The single JSONL results file")
    args = parser.parse_args()
    
    if os.path.exists(args.file):
        evaluate_file(args.file)
    else:
        print(f"File not found: {args.file}")
        print("Please ensure you have run some experiments with JSON saving enabled.")
