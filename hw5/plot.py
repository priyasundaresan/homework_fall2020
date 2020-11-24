import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
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
            if v.tag == value_tag:
                vals.append(v.simple_value)
    return np.array(vals)

train_log_dir = 'data'
train_logs = os.listdir(train_log_dir)

def plot_exp1():
    folders = [os.path.join(train_log_dir, f) for f in train_logs if 'q1_env' in f and ('Easy' in f or 'Medium' in f)]
    labels = ['PointmassEasy-Random', 'PointmassEasy-RND', 'PointmassMedium-Random', 'PointmassMedium-RND']
    for folder in folders:
        eval_returns = get_section_results(folder, 'Eval_AverageReturn')
        eval_stds= get_section_results(folder, 'Eval_StdReturn')
        plt.plot(np.arange(len(eval_returns)), eval_returns)
        plt.fill_between(np.arange(len(eval_returns)), eval_returns-eval_stds, eval_returns+eval_stds, alpha=0.5)
        plt.xlabel("Time")
        plt.ylabel('Returns')
    plt.title('Eval Average Returns over Time')
    plt.legend(labels)
    plt.savefig('q1-1.png')
    plt.clf()

def plot_exp2():
    folders = [os.path.join(train_log_dir, f) for f in train_logs if 'q1' in f]
    data = {"PointmassMedium":  (list(sorted([f for f in folders if 'Medium' in f])), ['cb', 'softmax', 'random', 'rnd']), \
            "PointmassHard":  (list(sorted([f for f in folders if 'Hard' in f])), ['cb', 'softmax', 'rnd'])}
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    fig.suptitle("Effect of Exploration in PointMass")
    i = 0
    for env in data:
        folders, labels = data[env]
        for label, folder in zip(labels, folders):
            eval_returns = get_section_results(folder, 'Eval_AverageReturn')
            eval_stds= get_section_results(folder, 'Eval_StdReturn')
            ax[i].plot(np.arange(len(eval_returns)), eval_returns)
            ax[i].fill_between(np.arange(len(eval_returns)), eval_returns-eval_stds, eval_returns+eval_stds, alpha=0.1)
        ax[i].set_title('%s'%env)
        ax[i].set_xlabel("Time")
        ax[i].set_ylabel('Returns')
        ax[i].legend(labels)
        i += 1
    plt.savefig('q1-2.png')
    plt.clf()

def plot_exp3():
    folders = [os.path.join(train_log_dir, f) for f in train_logs if 'q2_cql_PointmassMedium' in f \
                    or 'q2_dqn_PointmassMedium-v0_21-11-2020_23-36-54' in f or 'cql_scale_shift_PointmassMedium' in f]
    labels = ['DQN', 'CQL-Scale-Shift', 'CQL']
    for folder in folders:
        eval_returns = get_section_results(folder, 'Eval_AverageReturn')
        eval_stds= get_section_results(folder, 'Eval_StdReturn')
        plt.plot(np.arange(len(eval_returns)), eval_returns)
        plt.fill_between(np.arange(len(eval_returns)), eval_returns-eval_stds, eval_returns+eval_stds, alpha=0.1)
        plt.xlabel("Time")
        plt.ylabel('Returns')
    plt.title('Eval Average Returns over Time')
    plt.legend(labels)
    plt.savefig('q2-1.png')
    plt.clf()
    for folder in [folders[0],folders[-1]]:
        q_vals_data = get_section_results(folder, 'Exploitation_Data_q-values')
        q_vals_ood = get_section_results(folder, 'Exploitation_OOD_q-values')
        plt.plot(np.arange(len(q_vals_data)), q_vals_data)
        plt.plot(np.arange(len(q_vals_ood)), q_vals_ood)
    q_labels = []
    for l in [labels[0],labels[-1]]:
        q_labels.append('data-'+l)
        q_labels.append('ood-'+l)
    plt.legend(q_labels)
    plt.title('Q-Values')
    plt.savefig('q2-1-q_vals.png')
    plt.clf()


def plot_exp4():
    folders = list(sorted([os.path.join(train_log_dir, f) for f in train_logs if 'hw5_expl_q2_alpha' in f \
                    or 'q2_dqn_PointmassMedium-v0_21-11-2020_23-36-54' in f]))
    labels = ['alpha0.02', 'alpha0.1', 'alpha0.2', 'alpha0.3', 'alpha0.4', 'alpha0.5', 'DQN']
    for folder in folders:
        eval_returns = get_section_results(folder, 'Eval_AverageReturn')
        eval_stds= get_section_results(folder, 'Eval_StdReturn')
        plt.plot(np.arange(len(eval_returns)), eval_returns)
        plt.fill_between(np.arange(len(eval_returns)), eval_returns-eval_stds, eval_returns+eval_stds, alpha=0.2)
        plt.xlabel("Time")
        plt.ylabel('Returns')
    plt.title('Eval Average Returns over Time')
    plt.legend(labels)
    plt.savefig('q2-3.png')
    plt.clf()

#def plot_exp5():
#    folders = list(sorted([os.path.join(train_log_dir, f) for f in train_logs if 'numsteps' in f or 'q2_dqn_PointmassMedium-v0_21-11-2020_23-36-54' in f or 'hw5_expl_q2_cql_PointmassMedium-v0_21-11-2020_23-52-17' in f]))
#    print("here", folders)
#    labels = ['cql-10000', 'cql-1500', 'cql-5000', 'dqn-10000', 'dqn-1500', 'dqn-5000']
#    for folder in folders:
#        eval_returns = get_section_results(folder, 'Eval_AverageReturn')
#        eval_stds= get_section_results(folder, 'Eval_StdReturn')
#        plt.plot(np.arange(len(eval_returns)), eval_returns)
#        plt.fill_between(np.arange(len(eval_returns)), eval_returns-eval_stds, eval_returns+eval_stds, alpha=0.2)
#        plt.xlabel("Time")
#        plt.ylabel('Returns')
#    plt.title('Eval Average Returns over Time')
#    plt.legend(labels)
#    plt.savefig('q2-2.png')
#    plt.clf()

def plot_exp5():
    folders = list(sorted([os.path.join(train_log_dir, f) for f in train_logs if 'numsteps' in f or 'q2_dqn_PointmassMedium-v0_21-11-2020_23-36-54' in f or 'hw5_expl_q2_cql_PointmassMedium-v0_21-11-2020_23-52-17' in f]))
    data = {"CQL":  (list(sorted([f for f in folders if 'cql' in f])), [10000, 1500, 5000]), \
            "DQN":  (list(sorted([f for f in folders if 'dqn' in f])), [10000, 1500, 5000])}
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    fig.suptitle("Effect of # Exploration Steps in PointMass Medium")
    i = 0
    for env in data:
        folders, labels = data[env]
        for label, folder in zip(labels, folders):
            eval_returns = get_section_results(folder, 'Eval_AverageReturn')
            eval_stds= get_section_results(folder, 'Eval_StdReturn')
            ax[i].plot(np.arange(len(eval_returns)), eval_returns)
            ax[i].fill_between(np.arange(len(eval_returns)), eval_returns-eval_stds, eval_returns+eval_stds, alpha=0.1)
        ax[i].set_title('%s'%env)
        ax[i].set_xlabel("Time")
        ax[i].set_ylabel('Returns')
        ax[i].legend(labels)
        i += 1
    plt.savefig('q2-2.png')
    plt.clf()

def plot_exp6():
    folders = [os.path.join(train_log_dir, f) for f in train_logs if 'q3' in f or 'q1_env2_rnd_PointmassMedium' in f]
    data = {"PointmassMedium":  (list(sorted([f for f in folders if 'Medium' in f])), ['RND (Unsupervised)', 'CQL', 'DQN']), \
            "PointmassHard":  (list(sorted([f for f in folders if 'Hard' in f])), ['CQL', 'DQN'])}
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    fig.suptitle("Effect of Supervised Exploration in PointMass")
    i = 0
    for env in data:
        folders, labels = data[env]
        for label, folder in zip(labels, folders):
            eval_returns = get_section_results(folder, 'Eval_AverageReturn')
            eval_stds= get_section_results(folder, 'Eval_StdReturn')
            ax[i].plot(np.arange(len(eval_returns)), eval_returns)
            ax[i].fill_between(np.arange(len(eval_returns)), eval_returns-eval_stds, eval_returns+eval_stds, alpha=0.1)
        ax[i].set_title('%s'%env)
        ax[i].set_xlabel("Time")
        ax[i].set_ylabel('Returns')
        ax[i].legend(labels)
        i += 1
    plt.savefig('q3.png')
    plt.clf()

if __name__ == '__main__':
    plot_exp1()
    plot_exp2()
    plot_exp3()
    plot_exp4()
    plot_exp5()
    plot_exp6()
