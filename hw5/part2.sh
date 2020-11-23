#!/bin/bash

#python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_dqn --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0
#python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_cql --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 
#python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_cql_scale_shift --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --exploit_rew_shift 1 --exploit_rew_scale 100

#python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=5000 --offline_exploitation --cql_alpha=0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_5000
#python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=1500 --offline_exploitation --cql_alpha=0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_1500
#
#python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=5000 --offline_exploitation --cql_alpha=0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_5000
#python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=1500 --offline_exploitation --cql_alpha=0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_1500

#python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.02 --exp_name q2_alpha_0.02
python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --exp_name q2_alpha_0.1
python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.25 --exp_name q2_alpha_0.2
python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.35 --exp_name q2_alpha_0.3
python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.4 --exp_name q2_alpha_0.4
python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.5 --exp_name q2_alpha_0.5
