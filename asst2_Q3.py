#!/usr/bin/env python
# coding: utf-8

# reproduce 13.1

# In[14]:


import matplotlib.pyplot as plt
import numpy as np

from corridor import Corridor
from reinforce import Reinforce
from utils import feat_corr, pi_gen_corr, logpi_wrap_corr


# In[15]:


FIG_13_1_ALP_L = [2 ** (-k) for k in range(12, 15)]
FIG_13_1_N_EP = 1000
FIG_13_1_N_RUNS = 100
FIG_13_1_G = 1
FIG_13_1_THE_DIM = 2
FIG_13_1_OPT_REW = -11.6


# In[16]:


def run(ax, alg, alp_l, n_ep, n_runs, dash=True):
    for alp in alp_l:
        alg.a = alp
        print(f"[ALPHA={alp}]")
        tot_rew = np.zeros(n_ep)
        for seed in range(n_runs):
            print(f"[RUN #{seed}]")
            alg.reset()
            alg.seed(seed)
            tot_rew += np.array(alg.train(n_ep))
        plt.plot(tot_rew / n_runs, label=f'alpha=2 ** {np.log(alp) / np.log(2)}')
    if dash:
        plt.plot(np.zeros(n_ep) + FIG_13_1_OPT_REW, '--', label='v*(s_0)')


# In[17]:


def benchmark(alg, title, fn):
    fig, ax = plt.subplots()
    fig.suptitle(title)
    fig.set_size_inches(20, 14)

    xticks, yticks = np.linspace(0, 1000, 6), np.linspace(-90, -10, 9)
    def short_str(x): return str(x)[:3]
    xnames, ynames = map(short_str, xticks), map(short_str, yticks)
    run(ax, alg, FIG_13_1_ALP_L, FIG_13_1_N_EP, FIG_13_1_N_RUNS)
    plot_figure(ax, '', xticks, xnames, 'Episode', list(yticks) + [0], ynames,
              (f'Total\nReward\non episode\n(Averaged over\n' +
               f'{FIG_13_1_N_RUNS} runs)'), font=MED_FONT, labelpad=40,
              loc='upper right')
    save_plot(fn, dpi=100)
    plt.show()


# In[18]:


def fig_13_1():
    env = Corridor()

    alg = Reinforce(env, None, FIG_13_1_G, FIG_13_1_THE_DIM, pi_gen_corr,
                  logpi_wrap_corr(env, feat_corr), the_0=None)
    benchmark(alg, 'Figure 13.1', 'fig13.1')


# In[19]:


fig_13_1()


# In[ ]:




