#!/bin/bash

# NOTE, directly change lr in infrastructure/dqn_utils.py like so between runs:
#161 def lander_optimizer():
#162     return OptimizerSpec(
#163         constructor=optim.Adam,
#164         optim_kwargs=dict(
#165             lr=1,
#166         ),
#167         learning_rate_schedule=lambda epoch: <FIXME>,  # keep init learning rate
#168     )

python3 cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam3_lr1e-2
#python3 cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam2_lr1e-4
#python3 cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam1_lr1e-5
