#!/bin/bash

#python3 cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --use_rnd --unsupervised_exploration --exp_name q1_env1_rnd
#python3 cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --unsupervised_exploration --exp_name q1_env1_random
#python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --exp_name q1_env2_rnd
#python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --unsupervised_exploration --exp_name q1_env2_random

#python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --softmax_expl --unsupervised_exploration --exp_name q1_alg_med
#python3 cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --softmax_expl --unsupervised_exploration --exp_name q1_alg_hard


python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --count_based_expl --unsupervised_exploration --exp_name q1_alg_med_countbased
