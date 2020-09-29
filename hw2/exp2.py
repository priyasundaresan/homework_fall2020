#!/usr/bin/env python
import os

if __name__=="__main__":
    lrs = [1e-3, 1e-2, 1e-1]
    batch_sizes = [100, 1000]
    i = 1
    for lr in lrs:
        for bs in batch_sizes:
            print("Experiment: %d of %d, lr=%.4f, bs=%d"%(i, len(lrs)*len(batch_sizes), lr, bs))
            lr_str = str(lr)
            bs_str = str(bs)
            cmd = 'python3 cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b %s -lr %s -rtg \
                    --exp_name q2_b%s_r%s' % (bs_str, lr_str, bs_str, lr_str)
            os.system(cmd)
            i += 1
