#!/usr/bin/env python
# coding: utf-8

# (a). reproduce 6.2

# In[10]:


import matplotlib.pyplot as plt
from randomwalk import RandomWalk, NotSoRandomWalk, LEFT, RIGHT
from td_1 import OneStepTD
import numpy as np


# In[12]:


N_EP_EX_6_2 = 100
N_RUNS_EX_6_2 = 100
TRUE_VALUES_EX_6_2 = [1/6, 2/6, 3/6, 4/6, 5/6]
TD_STEPS_6_2 = [0.05, 0.1, 0.15]
MC_STEPS_6_2 = [0.01, 0.02, 0.03, 0.04] 
INIT_VAL_6_2 = 1/2


# In[ ]:





# In[4]:


def init_random_walk(init_value, step_size=None):
    env = RandomWalk()
    pi = {(a, s): 1.0 for s in env.states for a in env.moves} 
    V_0 = [init_value for s in env.states[:-1]] + [0]  # V = 0 for absorbing state
    V_init = {s: V_0[idx] for (idx, s) in enumerate(env.states)}
    alg = OneStepTD(env, V_init=V_init, step_size=step_size, gamma=1)
    return alg, pi


# In[14]:


def right_graph(fig, init_value, td_step_sizes, mc_step_sizes, remove_x_label=False, batch=False): 
    plt.title(f'V_init = {init_value}')
    alg, pi = init_random_walk(init_value)
    runs_dict = {alpha: np.zeros(N_EP_EX_6_2) for alpha in td_step_sizes + mc_step_sizes} 
    td_0 = alg.tabular_td_0 if not batch else alg.td_0_batch
    mc = alg.constant_step_size_mc if not batch else alg.constant_step_size_mc_batch
    to_compare_list = [(td_step_sizes, td_0), (mc_step_sizes, mc)]

    for (step_size_list, algorithm) in to_compare_list:
        for step_size in step_size_list:
            alg.step_size = step_size
            print(f"running step size {step_size}")
            for seed in range(N_RUNS_EX_6_2): 
                alg.reset()
                alg.env.seed(seed)
                err_l = []
                for nb_ep in range(N_EP_EX_6_2):
                    algorithm(pi, 1)
                    v_arr = np.array(alg.get_value_list()[:-1])
                    err_l.append(np.linalg.norm(v_arr-TRUE_VALUES_EX_6_2))
                runs_dict[step_size] += np.array(err_l)
    for key in runs_dict.keys():
        runs_dict[key] /= N_RUNS_EX_6_2

    if not remove_x_label:
        plt.xlabel('walks / episodes')
    plt.ylabel('empirical rms error averaged over states') 
    
    for key,err_run in runs_dict.items():
        (color, alg_name) = ('g','td') if key in td_step_sizes else ('r', 'mc')
        linewidth = max(int(100 * key) / 10 if key in td_step_sizes else int(200 * key) / 10, 10 / (len(runs_dict) * 10))
        linestyle = 'dashed' if key in [0.02, 0.03] else None
        plt.plot(err_run, color=color, label=alg_name + ' (a=' + str(key) + ')', linewidth=linewidth, linestyle=linestyle)

    plt.legend()


# In[16]:


def example_6_2():
  fig = plt.figure(dpi=300)
#   fig.suptitle('Example 6.2')
#   left_graph(fig, fig_id='121', init_value=INIT_VAL_6_2)
  right_graph(fig, INIT_VAL_6_2, TD_STEPS_6_2, MC_STEPS_6_2)
  plt.savefig('example6.2.png')
  plt.show()


# In[17]:


example_6_2()


# In[ ]:





# (b) reproduce 6.6

# In[ ]:





# In[24]:


from qlearning import QLearning
from sarsa import Sarsa
from cliff import TheCliff


# In[26]:


EX_6_6_N_EPS = 500
EX_6_6_YTICKS = [-100, -75, -50, -25]
EX_6_6_N_SEEDS = 10
EX_6_6_N_AVG = 50
EX_6_5_STEP_SIZE = 0.5
EX_6_5_EPS = 0.1


# In[29]:


def smooth_rewards(arr, to_avg=5):
    nb_rew = len(arr)
    new_arr = np.zeros(nb_rew - to_avg + 1) 
    for i in range(nb_rew - to_avg + 1):
        new_arr[i] = np.mean([arr[i + j] for j in range(to_avg)])
    return new_arr


# In[35]:


def example_6_6():
    fig, ax = plt.subplots(dpi=200) 
#     fig.suptitle(f'Example 6.6 (Averaged over {EX_6_6_N_SEEDS} seeds)')
    ax.set_xlabel('Episodes')
    ax.set_ylabel(f'(Average of last {EX_6_6_N_AVG}) sum of rewards during episodes')
    ax.set_yticks(EX_6_6_YTICKS)
    ax.set_ylim(bottom=min(EX_6_6_YTICKS))
    n_ep = EX_6_6_N_EPS
    env = TheCliff()
    qlearning_alg = QLearning(env, step_size=EX_6_5_STEP_SIZE, gamma=UNDISCOUNTED, eps=EX_6_5_EPS) 
    sarsa_alg = Sarsa(env, step_size=EX_6_5_STEP_SIZE, gamma=UNDISCOUNTED, eps=EX_6_5_EPS) 
    qlearning_rew = np.zeros(n_ep)
    sarsa_rew = np.zeros(n_ep)
    for seed in range(EX_6_6_N_SEEDS):
        print(f"seed={seed}")
        qlearning_alg.seed(seed)
        qlearning_rew += qlearning_alg.q_learning(n_ep)
        sarsa_alg.seed(seed)
        sarsa_rew += sarsa_alg.on_policy_td_control(n_ep, rews=True)
    plt.plot(smooth_rewards(qlearning_rew / EX_6_6_N_SEEDS, EX_6_6_N_AVG), color='r', label='Q learning')
    plt.plot(smooth_rewards(sarsa_rew / EX_6_6_N_SEEDS, EX_6_6_N_AVG), color='g', label='Sarsa')
    plt.legend()
    plt.savefig('example6.6.png')
    plt.show()


# In[36]:


example_6_6()


# In[ ]:




