import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import argparse

def get_section_results(folder, dagger=False):
    file = os.path.join(folder, os.listdir(folder)[0])
    eval_returns = []
    eval_stds = []
    expert_return = 0
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Eval_AverageReturn':
                eval_returns.append(v.simple_value)
            elif v.tag == 'Eval_StdReturn':
                eval_stds.append(v.simple_value)
            elif v.tag == 'Initial_DataCollection_AverageReturn':
                expert_return = v.simple_value
    if dagger:
        return eval_returns, eval_stds, expert_return
    return eval_returns, eval_stds

def plot_returns(hyperparam_values, returns, title, xlabel, ylabel):
    plt.scatter(hyperparam_values, returns)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_dagger(eval_returns, eval_stds, expert_return, title, xlabel, ylabel):
    #eval_returns = np.array(eval_returns[:-1])
    #eval_stds = np.array(eval_stds[:-1])
    eval_returns = np.array(eval_returns)
    eval_stds = np.array(eval_stds)
    iters = range(len(eval_returns))
    plt.plot(iters, eval_returns)
    plt.hlines(eval_returns[0], 0, len(iters))
    plt.hlines(expert_return, 0, len(iters), color ="green")
    plt.fill_between(iters, eval_returns-eval_stds, eval_returns+eval_stds, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(["DAgger", "BC", "Expert"])
    plt.show()

if __name__ == '__main__':
    data_dir = 'data'

    # Plotting BC
    #num_agent_train_steps_per_iter = [500,1000,2000,3000,3500,4000,7000,10000,20000]
    #network_paths = [os.path.join(data_dir, 'bc_ant_%dsteps'%i) for i in num_agent_train_steps_per_iter]
    #returns = [get_section_results(fn)[0][0] for fn in network_paths]
    #print(returns)
    #plot_returns(num_agent_train_steps_per_iter, returns, "Ant Environment: Num Agent Train Steps / Iter vs. Eval Returns", "Num Agent Train Steps Per Iteration", "Eval Mean Returns")

    # Plotting DAgger
    # Ant
    #dagger_path = os.path.join(data_dir, 'dagger_ant_2000steps_10iter')
    #returns, stds, expert_return = get_section_results(dagger_path, dagger=True)
    #plot_dagger(returns, stds, expert_return, 'Ant Environment: DAgger Returns vs. Iterations', 'Iterations', 'Evaluation Mean Returns')
    
    dagger_path = os.path.join(data_dir, 'dagger_walker_2000steps_10iter')
    returns, stds, expert_return = get_section_results(dagger_path, dagger=True)
    plot_dagger(returns, stds, expert_return, 'Walker Environment: DAgger Returns vs. Iterations', 'Iterations', 'Evaluation Mean Returns')
