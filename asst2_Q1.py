#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np


# In[8]:


def value_iteration(gamma=0.9, eps=0.5):
    gamma = gamma
    eps = eps
    
    n = 10000
    
    r = np.matrix([1, 2, 3]).T
    m = np.matrix([[3/4, 1/4, 0],
                  [1/4, 3/4, 0],
                  [0, eps, 1 - eps]])
    value_table = np.matrix(np.zeros(3)).T
    
    for i in range(n):
        old_value_table = np.copy(value_table)
        value_table = r + m * old_value_table
            
    return value_table, i+1
    


# In[3]:


gamma = 0.4
eps = 0.5
value_table, iterations = value_iteration(gamma=gamma, eps=eps)

print("With gamma={}, eps={}\nit ran for {} iterations\nthe optimal value is\n{}".format(gamma, 
                                                                                       eps,
                                                                                       iterations,
                                                                                       value_table))


# In[4]:


gamma = 0.999
eps = 0.5
value_table, iterations = value_iteration(gamma=gamma, eps=eps)

print("With gamma={}, eps={}\nit ran for {} iterations\nthe optimal value is\n{}".format(gamma, 
                                                                                       eps,
                                                                                       iterations,
                                                                                       value_table))


# In[5]:


gamma = 0.9
eps = 0.01
value_table, iterations = value_iteration(gamma=gamma, eps=eps)

print("With gamma={}, eps={}\nit ran for {} iterations\nthe optimal value is\n{}".format(gamma, 
                                                                                       eps,
                                                                                       iterations,
                                                                                       value_table))


# In[6]:


gamma = 0.999
eps = 0.01
value_table, iterations = value_iteration(gamma=gamma, eps=eps)

print("With gamma={}, eps={}\nit ran for {} iterations\nthe optimal value is\n{}".format(gamma, 
                                                                                       eps,
                                                                                       iterations,
                                                                                       value_table))


# In[ ]:





# In[ ]:




