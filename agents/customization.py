import torch
import torch.nn.utils.prune as prune
from .default import NormalNN
from .regularization import SI, EWC, EWC_online, L2
from .exp_replay import Naive_Rehearsal, GEM
from modules.criterions import BCEauto

import copy

import torch.nn as nn
import torch.nn.functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
from sklearn.metrics import f1_score
import scipy.stats
from .default import accumulate_acc

def init_zero_weights(m):
    with torch.no_grad():
        if type(m) == torch.nn.Linear:
            m.weight.zero_()
            m.bias.zero_()
        elif type(m) == torch.nn.ModuleDict:
            for l in m.values():
                init_zero_weights(l)
        else:
            assert False, 'Only support linear layer'


def NormalNN_reset_optim(agent_config):
    agent = NormalNN(agent_config)
    agent.reset_optimizer = True
    return agent


def NormalNN_BCE(agent_config):
    agent = NormalNN(agent_config)
    agent.criterion_fn = BCEauto()
    return agent


def SI_BCE(agent_config):
    agent = SI(agent_config)
    agent.criterion_fn = BCEauto()
    return agent


def SI_splitMNIST_zero_init(agent_config):
    agent = SI(agent_config)
    agent.damping_factor = 1e-3
    agent.reset_optimizer = True
    agent.model.last.apply(init_zero_weights)
    return agent


def SI_splitMNIST_rand_init(agent_config):
    agent = SI(agent_config)
    agent.damping_factor = 1e-3
    agent.reset_optimizer = True
    return agent


def EWC_BCE(agent_config):
    agent = EWC(agent_config)
    agent.criterion_fn = BCEauto()
    return agent


def EWC_mnist(agent_config):
    agent = EWC(agent_config)
    agent.n_fisher_sample = 60000
    return agent

def EWC_raf(agent_config):
    agent = EWC(agent_config)
    agent.n_fisher_sample = 12271
    return agent

def EWC_online_raf(agent_config):
    agent = EWC(agent_config)
    agent.n_fisher_sample = 12271
    agent.online_reg = True
    return agent

def EWC_online_mnist(agent_config):
    agent = EWC(agent_config)
    agent.n_fisher_sample = 60000
    agent.online_reg = True
    return agent


def EWC_online_empFI(agent_config):
    agent = EWC(agent_config)
    agent.empFI = True
    return agent


def EWC_zero_init(agent_config):
    agent = EWC(agent_config)
    agent.reset_optimizer = True
    agent.model.last.apply(init_zero_weights)
    return agent


def EWC_rand_init(agent_config):
    agent = EWC(agent_config)
    agent.reset_optimizer = True
    return agent


def EWC_reset_optim(agent_config):
    agent = EWC(agent_config)
    agent.reset_optimizer = True
    return agent


def EWC_online_reset_optim(agent_config):
    agent = EWC_online(agent_config)
    agent.reset_optimizer = True
    return agent


def Naive_Rehearsal_100(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 100
    return agent


def Naive_Rehearsal_200(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 200
    return agent


def Naive_Rehearsal_400(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 400
    return agent


def Naive_Rehearsal_1100(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 1100
    return agent


def Naive_Rehearsal_1400(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 1400
    return agent


def Naive_Rehearsal_4000(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 4000
    return agent


def Naive_Rehearsal_4400(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 4400
    return agent


def Naive_Rehearsal_5600(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 5600
    return agent


def Naive_Rehearsal_16000(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 16000
    return agent


def GEM_100(agent_config):
    agent = GEM(agent_config)
    agent.memory_size = 100
    return agent


def GEM_200(agent_config):
    agent = GEM(agent_config)
    agent.memory_size = 200
    return agent


def GEM_400(agent_config):
    agent = GEM(agent_config)
    agent.memory_size = 400
    return agent


def GEM_orig_1100(agent_config):
    agent = GEM(agent_config)
    agent.skip_memory_concatenation = True
    agent.memory_size = 1100
    return agent


def GEM_1100(agent_config):
    agent = GEM(agent_config)
    agent.memory_size = 1100
    return agent


def GEM_4000(agent_config):
    agent = GEM(agent_config)
    agent.memory_size = 4000
    return agent


def GEM_4400(agent_config):
    agent = GEM(agent_config)
    agent.memory_size = 4400
    return agent


def GEM_16000(agent_config):
    agent = GEM(agent_config)
    agent.memory_size = 16000
    return agent

class MagnitudePruning(NormalNN):
    def __init__(self, agent_config):
        super(MagnitudePruning, self).__init__(agent_config)
        self.prune_amount = agent_config.get('prune_amount', 0.0)
        self.prune_retrain_epochs = agent_config.get('prune_retrain_epochs', 1)

    def _get_parameters_to_prune(self):
        prunable = []
        trainable_names = {n.replace('.weight_orig', '.weight') for n, p in self.model.named_parameters() if p.requires_grad}
        for module_name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                param_name = f"{module_name}.weight" if module_name else "weight"
                if param_name in trainable_names:
                    prunable.append((module, 'weight'))
        return prunable

    def _retrain(self, train_loader, val_loader, epochs):
        if epochs <= 0:
            return
        self.init_optimizer()
        for epoch in range(epochs):
            self.model.train()
            for i, (inputs, targets, tasks) in enumerate(train_loader):
                if self.gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                self.update_model(inputs, targets, tasks)

    def learn_batch(self, train_loader, val_loader=None):
        super(MagnitudePruning, self).learn_batch(train_loader, val_loader)

        if self.prune_amount > 0.0:
            self.log(f'MagnitudePruning: Applying pruning with amount {self.prune_amount}')
            parameters_to_prune = self._get_parameters_to_prune()
            
            if len(parameters_to_prune) > 0:
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=self.prune_amount,
                )
                
                self.log(f'MagnitudePruning: Retraining for {self.prune_retrain_epochs} epochs')
                self._retrain(train_loader, val_loader, self.prune_retrain_epochs)
                
                for module, name in parameters_to_prune:
                    prune.remove(module, name)
                
                self.log('MagnitudePruning: Pruning completed.')

def MagnitudePruning_BCE(agent_config):
    agent = MagnitudePruning(agent_config)
    agent.criterion_fn = BCEauto()
    return agent

class BiasAwarePruning(NormalNN):
    def __init__(self, agent_config):
        super(BiasAwarePruning, self).__init__(agent_config)
        self.target_acc = agent_config.get('target_acc', 0.85)
        self.alpha = agent_config.get('alpha', 0.05)
        self.init_prune_rate = agent_config.get('init_prune_rate', 0.95)
        self.inc_prune_rate = agent_config.get('inc_prune_rate', 0.50)
        self.prune_retrain_epochs = agent_config.get('prune_retrain_epochs', 5)
        self.initial_state_dict = None
        self.baseline_ei = None
        self.baseline_acc = 0.0
        self.num_classes = list(self.config['out_dim'].values())[0] if isinstance(self.config['out_dim'], dict) else self.config['out_dim']
        self.chi2_threshold = scipy.stats.chi2.ppf(1 - self.alpha, self.num_classes - 1)
        self.log(f'BiasAwarePruning: Target Acc={self.target_acc}, Alpha={self.alpha}')

    def _reset_weights(self):
        """Resets the unpruned weights to their initial values."""
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    if prune.is_pruned(module):
                        orig_weight = self.initial_state_dict[f'{name}.weight_orig'] if f'{name}.weight_orig' in self.initial_state_dict else self.initial_state_dict[f'{name}.weight']
                        module.weight_orig.copy_(orig_weight)
                    else:
                        orig_weight = self.initial_state_dict[f'{name}.weight']
                        module.weight.copy_(orig_weight)

    def _get_parameters_to_prune(self):
        prunable = []
        trainable_names = {n.replace('.weight_orig', '.weight') for n, p in self.model.named_parameters() if p.requires_grad}
        for module_name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                param_name = f"{module_name}.weight" if module_name else "weight"
                if param_name in trainable_names:
                    prunable.append((module, 'weight'))
        return prunable

    def _evaluate_bias(self, val_loader):
        self.model.eval()
        correct_per_class = {}
        total_per_class = {}
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for i, (input, target, task) in enumerate(val_loader):
                if self.gpu:
                    input = input.cuda()
                    target = target.cuda()
                
                output = self.predict(input)
                if isinstance(output, dict):
                    output = output['All']
                
                _, predicted = torch.max(output, 1)
                
                total_samples += target.size(0)
                correct_mask = (predicted == target)
                total_correct += correct_mask.sum().item()
                
                for j in range(target.size(0)):
                    t = str(task[j].item() if isinstance(task[j], torch.Tensor) else task[j])
                    if t not in correct_per_class:
                        correct_per_class[t] = 0
                        total_per_class[t] = 0
                    
                    total_per_class[t] += 1
                    if correct_mask[j]:
                        correct_per_class[t] += 1
                    
        acc = total_correct / max(total_samples, 1)
        
        chi2_stat = 0.0
        for t in correct_per_class.keys():
            e_i = total_per_class[t] * acc
            if e_i > 0:
                o_i = correct_per_class[t]
                chi2_stat += ((o_i - e_i)**2) / e_i
                
        return acc, chi2_stat, correct_per_class

    def _retrain(self, train_loader, val_loader, epochs):
        self.init_optimizer()
        
        for epoch in range(epochs):
            self.model.train()
            self.log(f"BiasAwarePruning: Retraining Epoch {epoch + 1}/{epochs}")
            for i, (inputs, targets, tasks) in enumerate(train_loader):
                if i % 10 == 0:
                    self.log(f"  Batch {i}/{len(train_loader)}")
                if self.gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                self.update_model(inputs, targets, tasks)

    def learn_batch(self, train_loader, val_loader=None, global_val_loader=None):
        self.initial_state_dict = copy.deepcopy(self.model.state_dict())
        
        self.log('BiasAwarePruning: Step 1 - Training full network')
        super(BiasAwarePruning, self).learn_batch(train_loader, val_loader)
        
        self.converged_state_dict = copy.deepcopy(self.model.state_dict())
        
        bias_val_loader = global_val_loader if global_val_loader is not None else val_loader
        
        if bias_val_loader is not None:
            baseline_acc, _, correct_per_class = self._evaluate_bias(bias_val_loader)
            self.baseline_acc = baseline_acc
            self.baseline_ei = correct_per_class
            self.log(f'BiasAwarePruning: Baseline Acc = {self.baseline_acc:.4f}')
            self.log(f'BiasAwarePruning: Baseline expected correct per class e_i = {self.baseline_ei}')
            
            num_demo_classes = len(self.baseline_ei)
            if num_demo_classes > 1:
                self.chi2_threshold = scipy.stats.chi2.ppf(1 - self.alpha, num_demo_classes - 1)
                self.log(f'BiasAwarePruning: Updated Chi2 Threshold for {num_demo_classes} demographic classes = {self.chi2_threshold:.3f}')
        
        self.log('BiasAwarePruning: Step 2 - Coarse Pruning')
        best_p_coarse = 0.0
        low = 0.0
        high = self.init_prune_rate
        tolerance = 0.05
        first_iteration = True
        
        while (high - low) > tolerance:
            if first_iteration:
                mid = high
                first_iteration = False
            else:
                mid = (low + high) / 2.0
            self.log(f'BiasAwarePruning: Coarse pruning trying rate {mid:.3f}')
            
            for module, name in self._get_parameters_to_prune():
                if prune.is_pruned(module):
                    prune.remove(module, name)
            
            self.model.load_state_dict(self.converged_state_dict)
            
            params = self._get_parameters_to_prune()
            if len(params) > 0:
                prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=mid)
            
            self._reset_weights()
            self._retrain(train_loader, val_loader, self.prune_retrain_epochs)
            
            acc, chi2, _ = self._evaluate_bias(bias_val_loader)
            self.log(f'BiasAwarePruning: Coarse rate {mid:.3f} -> Acc: {acc:.4f}, Chi2: {chi2:.3f}')
            
            if acc >= self.target_acc and chi2 <= self.chi2_threshold:
                best_p_coarse = mid
                low = mid
                if mid == self.init_prune_rate:
                    break
            else:
                high = mid
                
        self.log(f'BiasAwarePruning: Best coarse prune rate found = {best_p_coarse:.3f}')
        
        for module, name in self._get_parameters_to_prune():
            if prune.is_pruned(module):
                prune.remove(module, name)
                
        self.model.load_state_dict(self.converged_state_dict)
        
        if best_p_coarse > 0:
            prune.global_unstructured(self._get_parameters_to_prune(), pruning_method=prune.L1Unstructured, amount=best_p_coarse)
            self._reset_weights()
            self._retrain(train_loader, val_loader, self.prune_retrain_epochs)
        
        # Make masks permanent before saving state so keys are clean (weight, not weight_orig/weight_mask)
        for module, name in self._get_parameters_to_prune():
            if prune.is_pruned(module):
                prune.remove(module, name)
            
        self.log('BiasAwarePruning: Step 3 - Fine Pruning')
        beta = self.inc_prune_rate
        current_prune_amount = best_p_coarse
        
        state_before_fine = copy.deepcopy(self.model.state_dict())
        
        while beta > 0.01:
            target_prune_amount = current_prune_amount + (1.0 - current_prune_amount) * beta
            self.log(f'BiasAwarePruning: Fine pruning with incremental rate beta = {beta:.3f} (Total {target_prune_amount:.3f})')
            
            # Remove any active masks from previous iteration
            for module, name in self._get_parameters_to_prune():
                if prune.is_pruned(module):
                    prune.remove(module, name)
            
            # Load clean state and apply new pruning
            self.model.load_state_dict(state_before_fine)
            prune.global_unstructured(self._get_parameters_to_prune(), pruning_method=prune.L1Unstructured, amount=target_prune_amount)
            
            acc, chi2, _ = self._evaluate_bias(bias_val_loader)
            self.log(f'BiasAwarePruning: Fine rate beta {beta:.3f} (Total {target_prune_amount:.3f}) -> Acc: {acc:.4f}, Chi2: {chi2:.3f}')
            
            if acc < self.target_acc or chi2 > self.chi2_threshold:
                self.log('BiasAwarePruning: Limits violated. Backtracking and reducing beta.')
                beta /= 2.0
            else:
                self.log('BiasAwarePruning: Fine pruning step successful.')
                # Make masks permanent and save new clean state
                for module, name in self._get_parameters_to_prune():
                    if prune.is_pruned(module):
                        prune.remove(module, name)
                current_prune_amount = target_prune_amount
                state_before_fine = copy.deepcopy(self.model.state_dict())
        self.log('BiasAwarePruning: Step 4 - Completed.')
        for module, name in self._get_parameters_to_prune():
            if prune.is_pruned(module):
                prune.remove(module, name) 

def BiasAwarePruning_BCE(agent_config):
    agent = BiasAwarePruning(agent_config)
    agent.criterion_fn = BCEauto()
    return agent

class SIMagnitudePruning(SI):
    def __init__(self, agent_config):
        super(SIMagnitudePruning, self).__init__(agent_config)
        self.prune_amount = agent_config.get('prune_amount', 0.0)
        self.prune_retrain_epochs = agent_config.get('prune_retrain_epochs', 1)

    def _get_parameters_to_prune(self):
        prunable = []
        trainable_names = {n.replace('.weight_orig', '.weight') for n, p in self.model.named_parameters() if p.requires_grad}
        for module_name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                param_name = f"{module_name}.weight" if module_name else "weight"
                if param_name in trainable_names:
                    prunable.append((module, 'weight'))
        return prunable

    def _retrain(self, train_loader, val_loader, epochs):
        if epochs <= 0:
            return
        self.init_optimizer()
        for epoch in range(epochs):
            self.model.train()
            for i, (inputs, targets, tasks) in enumerate(train_loader):
                if self.gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                self.update_model(inputs, targets, tasks)

    def learn_batch(self, train_loader, val_loader=None, is_last_task=True):
        super(SIMagnitudePruning, self).learn_batch(train_loader, val_loader)

        if not is_last_task:
            self.log('SIMagnitudePruning: Not the last task. Skipping pruning phase.')
            return

        if self.prune_amount > 0.0:
            self.log(f'SIMagnitudePruning: All tasks trained. Applying global unstructured pruning with amount {self.prune_amount}')
            parameters_to_prune = self._get_parameters_to_prune()
            
            if len(parameters_to_prune) > 0:
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=self.prune_amount,
                )
                
                self.log(f'SIMagnitudePruning: Retraining for {self.prune_retrain_epochs} epochs')
                self._retrain(train_loader, val_loader, self.prune_retrain_epochs)
                
                for module, name in parameters_to_prune:
                    prune.remove(module, name)
                
                self.log('SIMagnitudePruning: Pruning completed.')


def SIMagnitudePruning_BCE(agent_config):
    agent = SIMagnitudePruning(agent_config)
    agent.criterion_fn = BCEauto()
    return agent

class ProposedFramework(SI):
    def __init__(self, agent_config):
        super(ProposedFramework, self).__init__(agent_config)
        self.lambda_bias = agent_config.get('lambda_bias', 1.0)
        self.target_acc = agent_config.get('target_acc', 0.85)
        self.alpha = agent_config.get('alpha', 0.05)
        self.init_prune_rate = agent_config.get('init_prune_rate', 0.95)
        self.inc_prune_rate = agent_config.get('inc_prune_rate', 0.50)
        self.prune_retrain_epochs = agent_config.get('prune_retrain_epochs', 1)
        self.use_bias_objective = True
        self.use_bias_constraint = True
        self.pruning_score_mode = 'unified'
        
        self.mu_ref_list = []
        
        self.num_classes = list(self.config['out_dim'].values())[0] if isinstance(self.config['out_dim'], dict) else self.config['out_dim']
        self.chi2_threshold = scipy.stats.chi2.ppf(1 - self.alpha, self.num_classes - 1)
        self.log(f'ProposedFramework: Target Acc={self.target_acc}, Alpha={self.alpha}, Chi2 Threshold={self.chi2_threshold:.3f}')
        
        self.initial_state_dict = None
        self.baseline_ei = None

    def _bias_loss(self, loss_task):
        if (not self.use_bias_objective) or len(self.mu_ref_list) == 0:
            return torch.tensor(0.0, device=loss_task.device)

        mu_ref = sum(self.mu_ref_list) / len(self.mu_ref_list)
        return self.lambda_bias * ((loss_task - mu_ref) ** 2) / (mu_ref + 1e-8)

    def criterion(self, inputs, targets, tasks, regularization=True, **kwargs):
        loss = super(L2, self).criterion(inputs, targets, tasks, **kwargs)

        if regularization and len(self.regularization_terms) > 0:
            reg_losses = []
            for i, reg_term in self.regularization_terms.items():
                importance = reg_term['importance']
                task_param = reg_term['task_param']
                for n, p in self.params.items():
                    reg_losses.append((importance[n] * (p - task_param[n]) ** 2).sum())
            if len(reg_losses) > 0:
                loss += self.config['reg_coef'] * torch.stack(reg_losses).sum()
        return loss

    def update_model(self, inputs, targets, tasks):
        import time
        t0 = time.time()
        unreg_gradients = {}
        
        old_params = {}
        for n, p in self.params.items():
            old_params[n] = p.clone().detach()

        out = self.forward(inputs)
        loss_task = self.criterion(out, targets, tasks, regularization=False)
        
        L_bias = self._bias_loss(loss_task)
            
        loss_without_reg = loss_task + L_bias
        
        self.optimizer.zero_grad()
        loss_without_reg.backward()
        for n, p in self.params.items():
            if p.grad is not None:
                unreg_gradients[n] = p.grad.clone().detach()

        out2 = self.forward(inputs)
        loss_task2 = self.criterion(out2, targets, tasks, regularization=False)
        
        L_bias2 = self._bias_loss(loss_task2)

        loss_total = self.criterion(out2, targets, tasks, regularization=True) + L_bias2
        
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()
        
        for n, p in self.params.items():
            delta = p.detach() - old_params[n]
            if n in unreg_gradients.keys():  
                self.w[n] -= (unreg_gradients[n] * delta).detach()
                
        return loss_total.detach(), out2

    def _get_prunable_param_names(self):
        prunable_names = []
        for module_name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                prunable_names.append(f"{module_name}.weight" if module_name else "weight")
        return prunable_names

    def _compute_fairness_sensitivity(self, val_loader):
        b_scores = {n: torch.zeros_like(p) for n, p in self.params.items()}
        
        if (not self.use_bias_objective) or len(self.mu_ref_list) == 0:
            return b_scores 
        
        self.eval()
        for i, (inputs, targets, tasks) in enumerate(val_loader):
            if self.gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()
                
            out = self.forward(inputs)
            loss_task = self.criterion(out, targets, tasks, regularization=False)
            L_bias = self._bias_loss(loss_task)
            
            self.model.zero_grad()
            L_bias.backward()
            
            for n, p in self.params.items():
                if p.grad is not None:
                    b_scores[n] += (p.grad.abs() * p.abs() / len(val_loader)).detach()
                    
        self.train()
        return b_scores

    def _calculate_unified_importance_score(self, b_scores):
        S_scores = {}
        prunable_names = self._get_prunable_param_names()
        
        if len(self.regularization_terms) > 0:
            importance = self.regularization_terms[1]['importance']
        else:
            importance = {n: torch.zeros_like(p) for n, p in self.params.items()}
            
        all_m = []
        all_omega = []
        all_b = []
        
        for n, p in self.params.items():
            if n in prunable_names:
                all_m.append(p.abs().view(-1))
                all_omega.append(importance[n].view(-1))
                all_b.append(b_scores[n].view(-1))
            
        if len(all_m) > 0:
            all_m = torch.cat(all_m)
            all_omega = torch.cat(all_omega)
            all_b = torch.cat(all_b)
            
            min_m, max_m = all_m.min(), all_m.max()
            min_omega, max_omega = all_omega.min(), all_omega.max()
            min_b, max_b = all_b.min(), all_b.max()
        else:
            min_m = max_m = min_omega = max_omega = min_b = max_b = 0
            
        for n, p in self.params.items():
            if n in prunable_names:
                m_tilde = (p.abs() - min_m) / (max_m - min_m + 1e-8)
                if self.pruning_score_mode == 'magnitude':
                    S_scores[n] = 1.0 - m_tilde
                    continue

                omega_tilde = (importance[n] - min_omega) / (max_omega - min_omega + 1e-8)
                b_tilde = (b_scores[n] - min_b) / (max_b - min_b + 1e-8)
                
                S_scores[n] = (1.0 - m_tilde) * (1.0 - omega_tilde) * (1.0 - b_tilde)
                
        return S_scores

    def _apply_global_pruning(self, S_scores, amount):
        prunable_names = self._get_prunable_param_names()
        
        all_scores = []
        for n in prunable_names:
            if n in S_scores:
                all_scores.append(S_scores[n].view(-1))
            
        if len(all_scores) == 0: return
        
        all_scores = torch.cat(all_scores)
        nparams_toprune = int(round(amount * all_scores.nelement()))
        if nparams_toprune == 0: return
        
        topk = torch.topk(all_scores, k=nparams_toprune, largest=True)
        global_mask = torch.ones_like(all_scores, dtype=torch.bool)
        global_mask[topk.indices] = False
        
        masks = {}
        offset = 0
        for n in prunable_names:
            if n in S_scores:
                numel = S_scores[n].nelement()
                masks[n] = global_mask[offset:offset+numel].view(S_scores[n].shape).float()
                offset += numel
            
        for module_name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                param_name = 'weight'
                full_param_name = f"{module_name}.{param_name}" if module_name else param_name
                if full_param_name in masks:
                    prune.custom_from_mask(module, name=param_name, mask=masks[full_param_name])

    def _reset_weights(self):
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    if prune.is_pruned(module):
                        orig_weight = self.initial_state_dict[f'{name}.weight_orig'] if f'{name}.weight_orig' in self.initial_state_dict else self.initial_state_dict[f'{name}.weight']
                        module.weight_orig.copy_(orig_weight)
                    else:
                        orig_weight = self.initial_state_dict[f'{name}.weight']
                        module.weight.copy_(orig_weight)

    def _get_parameters_to_prune(self):
        prunable = []
        trainable_names = {n.replace('.weight_orig', '.weight') for n, p in self.model.named_parameters() if p.requires_grad}
        for module_name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                param_name = f"{module_name}.weight" if module_name else "weight"
                if param_name in trainable_names:
                    prunable.append((module, 'weight'))
        return prunable

    def _evaluate_bias(self, val_loader):
        self.model.eval()
        correct_per_class = {}
        total_per_class = {}
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for i, (input, target, task) in enumerate(val_loader):
                if self.gpu:
                    input = input.cuda()
                    target = target.cuda()
                
                output = self.predict(input)
                if isinstance(output, dict):
                    output = output['All']
                
                _, predicted = torch.max(output, 1)
                
                total_samples += target.size(0)
                correct_mask = (predicted == target)
                total_correct += correct_mask.sum().item()
                
                for j in range(target.size(0)):
                    t = str(task[j].item() if isinstance(task[j], torch.Tensor) else task[j])
                    if t not in correct_per_class:
                        correct_per_class[t] = 0
                        total_per_class[t] = 0
                    
                    total_per_class[t] += 1
                    if correct_mask[j]:
                        correct_per_class[t] += 1
                    
        acc = total_correct / max(total_samples, 1)
        
        chi2_stat = 0.0
        for t in correct_per_class.keys():
            e_i = total_per_class[t] * acc
            if e_i > 0:
                o_i = correct_per_class[t]
                chi2_stat += ((o_i - e_i)**2) / e_i
                
        return acc, chi2_stat, correct_per_class

    def _refresh_params(self):
        current_params = dict(self.model.named_parameters())
        for n in list(self.params.keys()):
            if n not in current_params:
                orig_n = n.replace('.weight', '.weight_orig')
                if orig_n in current_params:
                    self.params[n] = current_params[orig_n]
            else:
                self.params[n] = current_params[n]

    def _retrain(self, train_loader, val_loader, epochs):
        self.init_optimizer()
        self._refresh_params()
        for epoch in range(epochs):
            self.model.train()
            data_timer = Timer()
            batch_timer = Timer()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            acc = AverageMeter()

            data_timer.tic()
            batch_timer.tic()
            self.log(f'ProposedFramework: Retraining Epoch {epoch + 1}/{epochs}')
            self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
            
            for i, (inputs, targets, tasks) in enumerate(train_loader):
                data_time.update(data_timer.toc())
                
                if self.gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                loss, output = self.update_model(inputs, targets, tasks)
                
                inputs = inputs.detach()
                targets = targets.detach()
                
                acc = accumulate_acc(output, targets, tasks, acc)
                losses.update(loss, inputs.size(0))
                
                batch_time.update(batch_timer.toc())
                data_timer.toc()
                
                if ((self.config['print_freq']>0) and (i % self.config['print_freq'] == 0)) or (i+1)==len(train_loader):
                    self.log('[{0}/{1}]\t'
                          '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                          '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                          '{loss.val:.3f} ({loss.avg:.3f})\t'
                          '{acc.val:.2f} ({acc.avg:.2f})'.format(
                        i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, acc=acc))

    def learn_batch(self, train_loader, val_loader=None, global_val_loader=None, is_last_task=True):
        self.initial_state_dict = copy.deepcopy(self.model.state_dict())
        
        self.log('ProposedFramework: Phase 1 - Multi-Objective Regularised Training')
        super(ProposedFramework, self).learn_batch(train_loader, val_loader)
        
        self.converged_state_dict = copy.deepcopy(self.model.state_dict())
        
        self.eval()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            for i, (inputs, targets, tasks) in enumerate(train_loader):
                if self.gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                out = self.forward(inputs)
                loss = self.criterion(out, targets, tasks, regularization=False)
                total_loss += loss.item()
                batches += 1
        if batches > 0:
            self.mu_ref_list.append(total_loss / batches)
        self.train()
        
        if not is_last_task:
            self.log('ProposedFramework: Not the last task. Skipping pruning phase.')
            return
            
        bias_val_loader = global_val_loader if global_val_loader is not None else val_loader
        
        if bias_val_loader is not None:
            baseline_acc, _, correct_per_class = self._evaluate_bias(bias_val_loader)
            self.baseline_acc = baseline_acc
            self.baseline_ei = correct_per_class
            self.log(f'ProposedFramework: Baseline Acc = {self.baseline_acc:.4f}')
            self.log(f'ProposedFramework: Baseline expected correct per class e_i = {self.baseline_ei}')
            
            num_demo_classes = len(self.baseline_ei)
            if num_demo_classes > 1:
                self.chi2_threshold = scipy.stats.chi2.ppf(1 - self.alpha, num_demo_classes - 1)
                self.log(f'ProposedFramework: Updated Chi2 Threshold for {num_demo_classes} demographic classes = {self.chi2_threshold:.3f}')
            
            self.log('ProposedFramework: Phase 2 - Unified Importance Scoring')
            b_scores = self._compute_fairness_sensitivity(bias_val_loader)
            S_scores = self._calculate_unified_importance_score(b_scores)
            
            self.log('ProposedFramework: Phase 3 - Coarse Pruning')
            best_p_coarse = 0.0
            low = 0.0
            high = self.init_prune_rate
            tolerance = 0.05
            first_iteration = True
            
            while (high - low) > tolerance:
                if first_iteration:
                    mid = high
                    first_iteration = False
                else:
                    mid = (low + high) / 2.0
                self.log(f'ProposedFramework: Coarse pruning trying rate {mid:.3f}')
                
                for module, name in self._get_parameters_to_prune():
                    if prune.is_pruned(module):
                        prune.remove(module, name)
                
                self.model.load_state_dict(self.converged_state_dict)
                
                self._apply_global_pruning(S_scores, mid)
                self._reset_weights()
                self.log(f'Retrain for {self.prune_retrain_epochs} epochs')
                self._retrain(train_loader, val_loader, self.prune_retrain_epochs)

                acc, chi2, _ = self._evaluate_bias(bias_val_loader)
                self.log(f'ProposedFramework: Coarse rate {mid:.3f} -> Acc: {acc:.4f}, Chi2: {chi2:.3f}')
                
                bias_constraint_ok = (not self.use_bias_constraint) or chi2 <= self.chi2_threshold
                if acc >= self.target_acc and bias_constraint_ok:
                    best_p_coarse = mid
                    low = mid
                    if mid == self.init_prune_rate:
                        break
                else:
                    high = mid
                    
            self.log(f'ProposedFramework: Best coarse prune rate found = {best_p_coarse:.3f}')
            
            for module, name in self._get_parameters_to_prune():
                if prune.is_pruned(module):
                    prune.remove(module, name)
                    
            self.model.load_state_dict(self.converged_state_dict)
            
            if best_p_coarse > 0:
                self._apply_global_pruning(S_scores, best_p_coarse)
                self._reset_weights()
                self._retrain(train_loader, val_loader, self.prune_retrain_epochs)
            
            for module, name in self._get_parameters_to_prune():
                if prune.is_pruned(module):
                    prune.remove(module, name)
                
            self.log('ProposedFramework: Phase 3 - Fine Pruning')
            beta = self.inc_prune_rate
            current_prune_amount = best_p_coarse
            
            state_before_fine = copy.deepcopy(self.model.state_dict())
            
            while beta > 0.01:
                target_prune_amount = current_prune_amount + (1.0 - current_prune_amount) * beta
                self.log(f'ProposedFramework: Fine pruning with incremental rate beta = {beta:.3f} (Total {target_prune_amount:.3f})')
                
                for module, name in self._get_parameters_to_prune():
                    if prune.is_pruned(module):
                        prune.remove(module, name)
                
                self.model.load_state_dict(state_before_fine)
                self._apply_global_pruning(S_scores, target_prune_amount)

                acc, chi2, _ = self._evaluate_bias(bias_val_loader)
                self.log(f'ProposedFramework: Fine rate beta {beta:.3f} (Total {target_prune_amount:.3f}) -> Acc: {acc:.4f}, Chi2: {chi2:.3f}')
                
                bias_constraint_violated = self.use_bias_constraint and chi2 > self.chi2_threshold
                if acc < self.target_acc or bias_constraint_violated:
                    self.log('ProposedFramework: Limits violated. Backtracking and reducing beta.')
                    beta /= 2.0
                else:
                    self.log('ProposedFramework: Fine pruning step successful.')
                    for module, name in self._get_parameters_to_prune():
                        if prune.is_pruned(module):
                            prune.remove(module, name)
                    current_prune_amount = target_prune_amount
                    state_before_fine = copy.deepcopy(self.model.state_dict())
            
            self.log('ProposedFramework: Step 4 - Completed.')
            for module, name in self._get_parameters_to_prune():
                if prune.is_pruned(module):
                    prune.remove(module, name)

def ProposedFramework_BCE(agent_config):
    agent = ProposedFramework(agent_config)
    agent.criterion_fn = BCEauto()
    return agent

class ProposedFrameworkNoFairness(ProposedFramework):
    def __init__(self, agent_config):
        super(ProposedFrameworkNoFairness, self).__init__(agent_config)
        self.use_bias_objective = False
        self.use_bias_constraint = False
        self.log('Ablation: fairness components disabled.')


class ProposedFrameworkMagnitudeRanking(ProposedFramework):
    def __init__(self, agent_config):
        super(ProposedFrameworkMagnitudeRanking, self).__init__(agent_config)
        self.pruning_score_mode = 'magnitude'
        self.log('Ablation: pruning score mode set to magnitude-only.')


class ProposedFrameworkNoFairnessMagnitudeRanking(ProposedFramework):
    def __init__(self, agent_config):
        super(ProposedFrameworkNoFairnessMagnitudeRanking, self).__init__(agent_config)
        self.use_bias_objective = False
        self.use_bias_constraint = False
        self.pruning_score_mode = 'magnitude'
        self.log('Ablation: fairness path disabled and pruning score mode set to magnitude-only.')

