import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import argparse

def get_section_results(folder, value_tag):
    file = os.path.join(folder, os.listdir(folder)[0])
    eval_returns = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Eval_AverageReturn':
                eval_returns.append(v.simple_value)
    return eval_returns

def plot_returns(log_folders, value_tag, xlabel, ylabel, title):
    returns = [get_section_results(f, value_tag) for f in log_folders]
    for r in returns:
        plt.plot(np.arange(len(r)), r)
    plt.legend(log_folders)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

train_logs = os.listdir('data')

def plot_exp1():
    q1_folders = [os.path.join('data', f) for f in train_logs if 'q1' in f]
    sb_folders = [f for f in q1_folders if 'sb' in f]
    lb_folders = [f for f in q1_folders if 'lb' in f]
    plot_returns(sb_folders, 'Eval_AverageReturn', "Time", "Eval Average Return", "Exp1: Cart-Pole Small Batch")
    plt.clf()
    plot_returns(lb_folders, 'Eval_AverageReturn', "Time", "Eval Average Return", "Exp1: Cart-Pole Large Batch")

def plot_exp2():
    q2_folders = [os.path.join('data', f) for f in train_logs if 'q2' in f]
    plot_returns(q2_folders, "Eval_AverageReturn", "Time", "Eval Average Return", "Exp2: Inverted Pendulum Hyperparameter Search")

def plot_exp4():
    q4_folders = [os.path.join('data', f) for f in train_logs if 'q4' in f]
    plot_returns(q4_folders, 'Eval_AverageReturn', "Time", "Eval Average Return", "Exp4: Half Cheetah")

if __name__ == '__main__':
    #plot_exp1()
    plot_exp2()
    #plot_exp4()
