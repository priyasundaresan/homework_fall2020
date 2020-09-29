#!/usr/bin/env python
import os

if __name__=="__main__":
    lrs = [0.005, 0.01, 0.02]
    batch_sizes = [10000, 30000, 50000]
    i = 1
    for lr in lrs:
        for bs in batch_sizes:
            print("Experiment: %d of %d, lr=%.4f, bs=%d"%(i, len(lrs)*len(batch_sizes), lr, bs))
            lr_str = str(lr)
            bs_str = str(bs)
            cmd = 'python3 cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b %s -lr %s -rtg --nn_baseline \
                    --exp_name q4_search_b%s_lr%s_rtg_nnbaseline' % (bs_str, lr_str, bs_str, lr_str)
            os.system(cmd)
            i += 1
