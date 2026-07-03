# Bias Stability Regularised Learning

This repository contains a modified version of the continual learning bias mitigation framework from [CLForBiasMitigation](https://github.com/nchuramani/CLForBiasMitigation). It is used for RAF-DB experiments on bias stability regularised learning framework.

## Original Repository

This work is adapted from:

[https://github.com/nchuramani/CLForBiasMitigation](https://github.com/nchuramani/CLForBiasMitigation)

Citations:

```bibtex
@article{Churamani2022CL4BiasMitigation,
  author={Churamani, Nikhil and Kara, Ozgur and Gunes, Hatice},
  journal={IEEE Transactions on Affective Computing},
  title={{Domain-Incremental Continual Learning for Mitigating Bias in Facial Expression and Action Unit Recognition}},
  year={2022},
  pages={1-15},
  doi={10.1109/TAFFC.2022.3181033}
}

@inproceedings{Kara2021Towards,
  title={{Towards Fair Affective Robotics: Continual Learning for Mitigating Bias in Facial Expression and Action Unit Recognition}},
  author={Kara, Ozgur and Churamani, Nikhil and Gunes, Hatice},
  booktitle={{Workshop on Lifelong Learning and Personalization in Long-Term Human-Robot Interaction (LEAP-HRI), 16th ACM/IEEE International Conference on Human-Robot Interaction (HRI)}},
  year={2021}
}
```

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## RAF-DB Dataset

Place the RAF-DB dataset in the repository root as `RafDB/`.

Expected layout:

```text
BiasStabilityRegularisedLearning/
  RafDB/
    basic/
      Image/
        aligned/
      EmoLabel/
        list_partition_label.txt
      Annotation/
        manual/
```

## Running Experiments
### Hyperparameter Tuning

Run global tuning:

```bash
python tune_optuna.py --tune_stage global --n_trials $TRIALS --gpuid $GPUID --model_type resnet --model_name TVResNet18_pretrained_freeze --force_out_dim 7 --no_class_remap --offline_training
```

Run tuning for other baselines. Example:

```bash
python tune_optuna.py --tune_stage baseline --n_trials $TRIALS --gpuid $GPUID --model_type resnet --model_name TVResNet18_pretrained_freeze --force_out_dim 7 --agent_type customization --agent_name SIMagnitudePruning --no_class_remap --category gender --skip_unsure True
```

For the proposed method, first modify the relevant default values in `tune_proposed.py`, then run:

```bash
python tune_proposed.py --n_trials $TRIALS --gpuid $GPUID
```

### Training
Example:

```bash
python main.py --gpuid $GPUID --repeat $REPEAT --model_type resnet --model_name TVResNet18_pretrained_freeze --force_out_dim 7 --no_class_remap --optimizer Adam --batch_size 64 --lr 0.0013701119904548946 --weight_decay 4.3364101535633925e-05 --schedule 18 --agent_type customization --agent_name ProposedFramework --target_acc 0.8306796965850811 --reg_coef 23.407671342306866 --lambda_bias 2.6460518189656517 --init_prune_rate 0.8530534334439579 --inc_prune_rate 0.5663349248336013 --alpha 0.5663349248336013 --prune_retrain_epochs 5 --category gender --skip_unsure True

```

### Evaluation
```
python rafdb_eval_json.py
```