#!/usr/bin/env python
# coding: utf-8

# # Assignment 4: Continuous time models
# 
# ## [Scientific Computing with Python](http://scicompy.yoavram.com)
# ## Yoav Ram
# 
# # General instructions
# 
# 1. Do not remove any text or code cells; do not leave redundent print messages.
# 1. When instructed to implement a function, use the given function names and parameters lists; failure to do so may cause test functions to fail during grading.
# 1. When instructed to generate a plot, make sure that the plot is clear, that axes are propely labeled, and that the notebook is saved with the plot inline, so that the grader can see the plot without running the code. Make sure that you re-generate the plot if you changed the code!
# 1. Code lines with a triple comment `###` should not be removed or modified, they are used for automatic grading.
# 1. Note that there are 4 exercises and the last cell in the notebook says **end of assignment**; if you are missing anything please download the origianl file from the course website.
# 1. Your code should run within a reasonable time (a few minutes) and you should use idioms learned in class, e.g. array opreations, numba, multiprocessig.
# 1. Questions regarding the exercises should be posted to the course forum at the designated group (i.e. "assignment4"). You can post questions anonymously. You can also visit the Office Hours, but please do not email the course staff with questions about the exercise.
# 1. Intructions for submitting the exercise are on the [course website](https://scicompy.yoavram.com/assignments).

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import numba
import scipy.stats
from scipy.integrate import solve_ivp
import seaborn as sns
sns.set_context('notebook')
red, blue, green, purple, orange, yellow = sns.color_palette('Set1', 6)


# # Background
# 
# Recall our work on the predator-prey model:

# In[ ]:


def dxydt(t, xy, b, h, ϵ, d):
    x, y = xy
    dx = b * x - h * x * y
    dy = ϵ * h * x * y - d * y
    return np.array([dx, dy])


# In[ ]:


b = 1
h = 0.005
ϵ = 0.8
d = 0.6

steps = 50000# number integration steps
XY = np.empty((2, steps))# population array
XY[:,0] = 50, 100# initial population sizes
dt = 0.001# time step for integration

for t_ in range(1, steps):
    XY[:,t_] = XY[:, t_-1] + dxydt(t_, XY[:, t_-1], b, h, ϵ, d) * dt
X, Y = XY
T = np.arange(0, steps*dt, dt)


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(8, 4))

ax = axes[0]
ax.plot(T, X, label='prey')
ax.plot(T, Y, label='predator')
ax.set_xlabel('Time')
ax.set_ylabel('Count')
ax.legend();

ax = axes[1]
ax.plot(X, Y, lw=0.5)
ax.set_xlabel('Prey')
ax.set_ylabel('Predator')
ax.set(xlim=(0, None), ylim=(0, None))

fig.tight_layout()


# We talked about how these cycles only *seem* stable, but really, they are prone to extinctions due to stochastic events, leading to either exitinction of both species or extinction of the predator and explosion of the prey population.
# 
# In this assignment you will implement a stochastic simulation of the dynamics using Gillespie's algorithm, which we covered in lecture 10.

# # Ex 1
# 
# **Write a table of reactions and their rates for the predator-prey dynamics**
# Note that this is not a coding exercise, but rather a written exercise: use Markdown/Latex to create the table.
# 
# $x$ denote the prey, $y$ denotes the predator; use the same rates as used in the deterministic model: $b$, $h$, $\epsilon$, and $d$.
# 
# Note that $h$ describes the rate at which predators hunts prey, whereas $\epsilon$ described the rate at which predators convert prey mass into predator mass.
# We didnt have a scenario like that in the molecular dynamics case and it requires a bit of sophistication.

# | Reaction   | State            |
# |------------|------------------|
# | x -> x + 1 | bx                |
# | x -> x - 1 | h*y*x              |
# | y -> y + 1 | $\epsilon$ * h * x *y |
# | y -> y - 1 | d*y                |

# **WRITE THE REACTION TABLE HERE**

# # Ex 2
# 
# **Write a Gillespie simulation for the predator-prey dynamics.**
# 
# **Run a single simulation and plot it together with the deterministic dynamics.**
# 
# Plot both prey and predator population sizes vs time, as well as a phase plot of the predator vs prey.
# 
# Note 1: once the predators are extinct there is no more use to continue running the simulation as the prey will just grow exponentiallt according to $dx/dt = bx$.
# 
# Note 2: since this is a stochastic simulation, your results may differ from mine.

# In[ ]:


@numba.jit # 2-fold faster
def get_rates(b, h, y, ϵ, x, d):
    return np.array([
        b * x,     # Prey production
        h * y * x,   # Prey degradation
        ϵ * ℎ * 𝑥 * y, # Predator production
        d * y      # Predator degradation
    ])

@numba.jit # 2-fold faster
def draw_time(rates):
    total_rate = rates.sum()
    return np.random.exponential(1/total_rate)

# @numba.jit # jit causes errors with multinomial
def draw_reaction(rates):
    rates /= rates.sum()
    return np.random.multinomial(1, rates).argmax()


# In[ ]:


updates = np.array([
    [1, 0],  # Prey production
    [-1, 0], # Prey degradation
    [0, 1],  # Predator production
    [0, -1]  # Predator degradation
])


# In[ ]:


def gillespie_step(b, h, y, ϵ, x, d):
    rates = get_rates(b, h, y, ϵ, x, d)
    Δt = draw_time(rates)
    ri = draw_reaction(rates)
    Δx, Δy = updates[ri]
    return Δt, Δx, Δy


# In[ ]:


def gillespie_ssa(b ,h ,ϵ, d,tmax=10 ,y0=50 , x0=100,t0=0, t_steps=1000 ):
    times = np.linspace(t0, tmax, t_steps) # recording times: time points in which to record the state
    states = np.empty((updates.shape[1], t_steps), dtype=int) # recorded states
    
    # init
    t = t0
    x, y = x0, y0
    Δx, Δy = 0, 0
    # loop over recording times
    for i, next_t in enumerate(times):
        # simulate until next recording time
        while t < next_t:
            if y == 0:
                if x == 0:
                    Δt, Δx, Δy = 1, 0, 0
                else:
                    Δt, Δx, Δy = 1, 1, 0
            else:
                Δt, Δx, Δy = gillespie_step(b, h, y, ϵ, x, d)
            t, x, y = t+Δt, x+Δx, y+Δy

        # record the previous state for the time point we just passed
        states[:, i] = x - Δx, y - Δy
    # return array equivalent to [times, Prey, Predator] for t in times]
    return np.concatenate((times.reshape(1, -1), states), axis=0)

def plot_xy(t, x, y, label='', ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(t, x, lw=3, label='x ' + label)
    ax.plot(t, y, lw=3, label='y ' + label)
    ax.set_xlabel('Time')
    ax.set_ylabel('Prey')
    ax.legend()
    return ax

t, x, y = gillespie_ssa(b, h, ϵ,d, tmax=50, x0=50, y0=100)


# In[15]:


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
t, x, y = gillespie_ssa(b, h, ϵ,d, tmax=50, x0=50, y0=100)
ax = axes[0]
ax.plot(T, X, label='ODE')
ax.plot(t, x, label='SSA')
ax.set_xlabel('Time')
ax.set_ylabel('Prey')
ax.legend();
ax = axes[1]
ax.plot(T, Y, label='ODE')
ax.plot(t, y, label='SSA')
ax.set_xlabel('Time')
ax.set_ylabel('Predator')
ax.legend();
ax = axes[2]
ax.plot(X, Y, label='ODE')
ax.plot(x, y, label='SSA')
ax.set_xlabel('Prey')
ax.set_ylabel('Predator')
ax.set(xlim=(0, None), ylim=(0, None))

ax.legend();



fig.tight_layout()


# # Ex 3
# 
# **Calculate the extinction probability of the predators in the first 50 days (assuming `t` is in days) and plot it as a function of $h$ the hunting probability.**
# 
# The extinction probability is the probability that the predators populations size reaches zero.
# To do that, you will have to run many simulations for the same parameters and check what is the fraction that finished with zero predators.
# 
# Think: How many replications should you use per parameter set?
# Remember that the standard error of the mean generally decreases like the root of the number of observations ($\sqrt{n}$).
# 
# When choosing the number of $h$ values, think if you want to use `np.linspace` or `np.logspace`, or maybe draw random values (from which distribution?) and how many points you should use.
# 
# Note that this exercise will require running many simulations; if we estimate the probability from just 100 simulations, and plot against just 10 values of $h$, we still need to run 1000 simulations.
# 
# There are several ways to attack this, and they are not mutually exclusive:
# 1. optimize the simulation code
# 1. run in parallel on multiple cores on your own machine (see end of lecture 10).
# 1. use cloud computing on your own (see end of lecture 10).
# 1. use the [course's cloud computing](https://scicompy-jupyter.yoavram.com) as explained in class. Remember that this is a shared resource for all course participants, so try to use it when it is free.
# 
# At any case, make sure to save your simulation results to files so that you can reload them again and change the analysis or plot the figure again (but do not include these files in the assignment submission).

# In[ ]:


from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings


# In[ ]:


num_of_cpus=cpu_count()
reps = 7000

def simulation(h):
  pred_extinct_counter = 0
  args = (b ,h ,ϵ, d)
  with ProcessPoolExecutor(num_of_cpus) as exec:
    futs = [exec.submit(gillespie_ssa, *args) for i in range(reps)]

    for cur_finished_thread in as_completed(futs):
      if cur_finished_thread.exception():
        warnings.warn(str(cur_finished_thread.exception()))
      else:
        result_gillespie_ssa = cur_finished_thread.result()
        if result_gillespie_ssa[2][-1] == 0:
          pred_extinct_counter = pred_extinct_counter + 1
  return pred_extinct_counter/reps


# In[ ]:


number_of_reps = 30
h_arr = np. linspace(0.000001,1,number_of_reps)
results = np.empty((2, number_of_reps))
counter = 0
for h in h_arr:
  results[:,counter]= (h,simulation(h))
  counter = counter + 1

plt.plot(results[0], results[1],label='Predators extinction probability')
plt.xlabel('h')
plt.ylabel('Predators probability to be extincted')
plt.legend();


# # Ex 4
# 
# Here's data from a real predator-prey system -- the hare and lynx system:

# In[16]:


data = np.fromstring("""0 20 10 
2 55 15 
4 65 55 
6 95 60 
8 55 20 
10 5 15 
12 15 10 
14 50 60 
16 75 60 
18 20 10 
20 25 5 
22 50 25 
24 70 40 
26 30 25 
28 15 5 
""", sep=' ', dtype=int)
data = data.reshape(-1, 3)
data[:, 1] *= 1000
data[:, 2] *= 100
print(data)


# The first column is years, the second is hare population size, the third is lynx population size.
# 
# Here is a plot the data.

# In[17]:


t, hare, lynx = data.T
plt.plot(t, hare, '-o', label='hare')
plt.plot(t, lynx, '-o', label='lynx')
plt.yscale('log')
plt.xlabel('years')
plt.ylabel('count')
plt.legend()

sns.despine()


# We can see the cycles that we discussed in lecture 10.
# 
# **Now use ABC to fit a model to the data**, just as we did in lecture 11.
# 
# 
# Do not use summary statistics, rather, calculate the MSE (mean squared error) between the data and the simulations.
# The `ABCSMC` constructor's third argument is a distance function that accepts the simulation dict as a first argument and the real dict as a second argument (the real dict is the one you give to the `new` method later on). See [example](https://pyabc.readthedocs.io/en/latest/examples/parameter_inference.html).
# 
# **Tips**
# - for `b` and `d` choose an `expon(1)` prior distribution
# - for `h` and `ϵ` choose `uniform(0, 1)` prior distribution
# - if you get division by zero errors, the population has probably gone extinct, you should make sure you stop the simulation when the population is extinct
# - running time may vary, but will take at least several minutes

# In[ ]:


import os
import tempfile
from pyabc import ABCSMC, RV, Distribution
from pyabc.visualization import plot_kde_1d, plot_kde_2d


# In[20]:


def model(params):
    # run a single simulation
    sim_res = gillespie_ssa(params.b, params.h, params.ϵ, params.d, t_steps=15)
    return dict( ###
        times = sim_res[0] , hare = sim_res[1], lynx = sim_res[2] 
    ) ###
    
prior = Distribution( ###
    # set the parameter prior distributions here
    h=RV("uniform", 0, 1),
    ϵ=RV("uniform", 0, 1),
    b=RV("expon", 1),
    d=RV("expon", 1)
    ) ###

def mse(x, y): ###
    # compute the distance between data and simulation (distance is symmetric so it doesn't matter which is x and which is y)
    x_tuple = [x['hare'], x['lynx']]
    y_tuple = [y['hare'], y['lynx']]
    return (np.subtract(x_tuple, y_tuple)**2).mean()

# create the ABC object and init it with the new method
abc = ABCSMC(model, prior, mse)

abc.new("sqlite:///" + os.path.join(tempfile.gettempdir(), "newfile.db"), {"times": t, "hare": hare, "lynx": lynx})


# In[21]:


#%%time ###
minimum_epsilon = 1259976732.7833333
max_nr_populations = 3
history = abc.run(minimum_epsilon=minimum_epsilon, max_nr_populations=max_nr_populations)


# We now plot the approximated posterior distributions of the model parameters.

# In[22]:


params, weights = history.get_distribution(0)
params.head()


# In[23]:


fig, axes = plt.subplots(1, 4, figsize=(12, 4))

for param_name, ax in zip(params.columns, axes.flat):
    sns.distplot(params[param_name], hist_kws=dict(weights=weights), ax=ax)
    ax.axvline(params[param_name].values @ weights, color=red, ls='--')
    ax.set(xlabel=param_name, ylabel='Posterior')

fig.tight_layout()


# **end of assignment**
