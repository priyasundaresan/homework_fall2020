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
    file = os.path.join(folder, os.listdir(folder)[0])
    time = []
    eval_returns = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                time.append(v.simple_value)
            if v.tag == value_tag:
                eval_returns.append(v.simple_value)
    return time[-1], eval_returns

train_log_dir = 'cs285/data'
train_logs = os.listdir(train_log_dir)

def plot_exp1():
    q1_folder = [os.path.join(train_log_dir, f) for f in train_logs if 'q1' in f][0]
    tavg, avg_returns = get_section_results(q1_folder, 'Train_AverageReturn')
    tmax, max_returns = get_section_results(q1_folder, 'Train_BestReturn')
    plt.plot(np.arange(len(avg_returns))*(tavg/len(avg_returns)), avg_returns)
    plt.plot(np.arange(len(max_returns))*(tmax/len(max_returns)), max_returns)
    plt.legend(['average', 'best'])
    plt.xlabel("Time")
    plt.ylabel('Returns')
    plt.title('Train Returns over Time: Ms. PacMan')
    plt.savefig('q1.png')
    plt.clf()

def plot_exp2():
    q2_folders = [os.path.join(train_log_dir, f) for f in train_logs if 'q2' in f]
    q2_dqn_folders = [f for f in q2_folders if 'q2_dqn' in f]
    q2_ddqn_folders = [f for f in q2_folders if 'q2_double' in f]
    dqn_returns = [get_section_results(f, 'Train_AverageReturn') for f in q2_dqn_folders]
    avg_dqn_returns = np.mean(np.array([i[1] for i in dqn_returns]), axis=0)
    ddqn_returns = [get_section_results(f, 'Train_AverageReturn') for f in q2_ddqn_folders]
    avg_ddqn_returns = np.mean(np.array([i[1] for i in ddqn_returns]), axis=0)
    t_avg_dqn = dqn_returns[0][0]
    plt.plot(np.arange(len(avg_dqn_returns))*(t_avg_dqn/len(avg_dqn_returns)), avg_dqn_returns)
    plt.plot(np.arange(len(avg_ddqn_returns))*(t_avg_dqn/len(avg_dqn_returns)), avg_ddqn_returns)
    plt.legend(['dqn', 'double dqn'])
    plt.xlabel("Time")
    plt.ylabel('Returns')
    plt.title('Train Returns (averaged across 3 trials): LunarLander')
    plt.savefig('q2.png')
    plt.clf()

def plot_exp3():
    q2_folders = [os.path.join(train_log_dir, f) for f in train_logs if 'q2' in f]
    q2_dqn_folders = [f for f in q2_folders if 'q2_dqn' in f]
    dqn_returns = [get_section_results(f, 'Train_AverageReturn') for f in q2_dqn_folders]
    avg_dqn_returns = np.mean(np.array([i[1] for i in dqn_returns]), axis=0)
    t_avg_dqn = dqn_returns[0][0]
    plt.plot(np.arange(len(avg_dqn_returns))*(t_avg_dqn/len(avg_dqn_returns)), avg_dqn_returns)
    q3_folders = [os.path.join(train_log_dir, f) for f in train_logs if 'q3' in f]
    dqn_returns = [get_section_results(f, 'Train_AverageReturn') for f in q3_folders]

    for t,r in dqn_returns:
        plt.plot(np.arange(len(r))*(t/len(r)),r)
    plt.xlabel("Time")
    plt.ylabel('Returns')
    plt.title('DQN Train Average Returns vs. Time: LunarLander')
    plt.legend(['1e-3', '1e-5', '1e-4', '1e-2'])
    plt.savefig('q3.png')
    plt.clf()

def plot_exp4():
    q4_folders = [os.path.join(train_log_dir, f) for f in train_logs if 'q4' in f]
    pprint.pprint(q4_folders)
    returns = [get_section_results(f, 'Train_AverageReturn') for f in q4_folders]
    for t,r in returns:
        plt.plot(np.arange(len(r))*(t/len(r)),r)
    plt.xlabel("Time")
    plt.ylabel('Returns')
    plt.legend(['100target_1grad', '10target_10grad', '1target_1grad', '1target_100grad'])
    plt.title('Train Average Returns vs. # Target/Gradient Updates: CartPole')
    plt.savefig('q4.png')
    plt.clf()

def plot_exp5():
    q5_folders = [os.path.join(train_log_dir, f) for f in train_logs if 'q5_100_1' in f]
    returns = [get_section_results(f, 'Eval_AverageReturn') for f in q5_folders]
    (t_cheetah,r_cheetah), (t_pendulum, r_pendulum) = returns
    plt.plot(np.arange(len(r_cheetah))*(t_cheetah/len(r_cheetah)), r_cheetah)
    plt.xlabel("Time")
    plt.ylabel('Avg Eval Returns')
    plt.title("Eval Average Returns vs. Time: HalfCheetah")
    plt.legend(['100target_1grad'])
    plt.savefig('q5a.png')
    plt.clf()
    plt.plot(np.arange(len(r_pendulum))*(t_pendulum/len(r_pendulum)), r_pendulum)
    plt.xlabel("Time")
    plt.ylabel('Avg Eval Returns')
    plt.title("Eval Average Returns vs. Time: InvertedPendulum")
    plt.legend(['100target_1grad'])
    plt.savefig('q5b.png')
    plt.clf()

if __name__ == '__main__':
    plot_exp1()
    plot_exp2()
    plot_exp3()
    plot_exp4()
    plot_exp5()
