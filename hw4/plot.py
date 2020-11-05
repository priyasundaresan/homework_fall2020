import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import argparse
import pprint

def get_section_results(folder, value_tag):
    file = os.path.join(folder, [f for f in os.listdir(folder) if '.deathstar' in f][0])
    #time = []
    vals = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            #if v.tag == 'Train_EnvstepsSoFar':
            #    time.append(v.simple_value)
            if v.tag == value_tag:
                vals.append(v.simple_value)
    return np.array(vals)

train_log_dir = 'data'
train_logs = os.listdir(train_log_dir)

def plot_exp2():
    folder = [os.path.join(train_log_dir, f) for f in train_logs if 'q2' in f][0]
    avg_returns = get_section_results(folder, 'Train_AverageReturn')
    eval_returns = get_section_results(folder, 'Eval_AverageReturn')
    plt.scatter(np.arange(len(avg_returns)), avg_returns)
    plt.scatter(np.arange(len(eval_returns)), eval_returns)
    plt.legend(['Train Average Return', 'Eval Average Return'])
    plt.xlabel("Time")
    plt.ylabel('Returns')
    plt.title('Returns over Time: %s'%'Obstacles')
    plt.savefig('q2.png')
    plt.clf()

def plot_exp3():
    folders = [os.path.join(train_log_dir, f) for f in train_logs if 'q3' in f]
    tasks = ['Reacher', 'Obstacles', 'Cheetah']
    for task, folder in zip(tasks, folders):
        avg_returns = get_section_results(folder, 'Train_AverageReturn')
        avg_stds= get_section_results(folder, 'Train_StdReturn')
        eval_returns = get_section_results(folder, 'Eval_AverageReturn')
        eval_stds= get_section_results(folder, 'Eval_StdReturn')
        plt.plot(np.arange(len(avg_returns)), avg_returns)
        plt.fill_between(np.arange(len(avg_returns)), avg_returns-avg_stds, avg_returns+avg_stds, alpha=0.5)
        plt.plot(np.arange(len(eval_returns)), eval_returns)
        plt.fill_between(np.arange(len(eval_returns)), eval_returns-eval_stds, eval_returns+eval_stds, alpha=0.5)
        plt.legend(['Train Average Return', 'Eval Average Return'])
        plt.xlabel("Time")
        plt.ylabel('Returns')
        plt.title('Returns over Time: %s'%task)
        plt.savefig('q3-%s.png'%task)
        plt.clf()

def plot_exp4():
    folders = [os.path.join(train_log_dir, f) for f in train_logs if 'q4' in f]
    data = {
    'Ensembles': (list(sorted([f for f in folders if 'ensemble' in f])), ['1 ensemble', '3 ensemble', '5 ensemble']), 
    'Number of Sequences': (list(sorted([f for f in folders if 'numseq' in f])), ['100 sequences', '1000 sequences']),
    'Planning Horizon': (list(sorted([f for f in folders if 'horizon' in f])), ['horizon 5', 'horizon 15', 'horizon 30'])
    }
    for k in data:
        hyperparam = k
        folders, labels = data[k]
        fig, ax = plt.subplots(1, len(folders), figsize=(20, 5))
        i = 0
        fig.suptitle("Effect of %s on Performance on Reacher Task"%hyperparam)
        for label, folder in zip(labels, folders):
            print(label, folder)
            avg_returns = get_section_results(folder, 'Train_AverageReturn')
            avg_stds= get_section_results(folder, 'Train_StdReturn')
            eval_returns = get_section_results(folder, 'Eval_AverageReturn')
            eval_stds= get_section_results(folder, 'Eval_StdReturn')
            ax[i].plot(np.arange(len(avg_returns)), avg_returns)
            ax[i].fill_between(np.arange(len(avg_returns)), avg_returns-avg_stds, avg_returns+avg_stds, alpha=0.5)
            ax[i].plot(np.arange(len(eval_returns)), eval_returns)
            ax[i].fill_between(np.arange(len(eval_returns)), eval_returns-eval_stds, eval_returns+eval_stds, alpha=0.5)
            ax[i].legend(['Train', 'Eval'])
            ax[i].set_title('%s'%label)
            ax[i].set_xlabel("Time")
            ax[i].set_ylabel('Returns')
            i += 1
        plt.savefig('q4-%s.png'%hyperparam)
        plt.clf()

if __name__ == '__main__':
    #plot_exp2()
    #plot_exp3()
    plot_exp4()
