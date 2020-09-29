#!/usr/bin/env python
import os
import _thread as thread

if __name__ == '__main__':
    #b = 30000
    r = 0.02
#    os.system('python3 cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 150 -l 2 -s 32 -b 30000 -lr 0.02 --exp_name q4_b30000_r0.02')
#    os.system('python3 cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 150 -l 2 -s 32 -b 30000 -lr 0.02 -rtg --exp_name q4_b30000_r0.02_rtg')
#    os.system('python3 cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 150 -l 2 -s 32 -b 30000 -lr 0.02 --nn_baseline --exp_name q4_b30000_r0.02_nnbaseline')
#    os.system('python3 cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 150 -l 2 -s 32 -b 30000 -lr 0.02 -rtg --nn_baseline --exp_name q4_b30000_r0.02_rtg_nnbaseline')
#
    b = 50000
    os.system('python3 cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 150 -l 2 -s 32 -b 50000 -lr 0.02 --exp_name q4_b50000_r0.02')
    os.system('python3 cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 150 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --exp_name q4_b50000_r0.02_rtg')
    os.system('python3 cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 150 -l 2 -s 32 -b 50000 -lr 0.02 --nn_baseline --exp_name q4_b50000_r0.02_nnbaseline')
    os.system('python3 cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 150 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline --exp_name q4_530000_r0.02_rtg_nnbaseline')
