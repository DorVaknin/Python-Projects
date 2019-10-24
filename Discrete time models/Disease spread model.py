#!/usr/bin/env python
# coding: utf-8



# In[4]:


import random

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
sns.set_context('notebook')
colors = sns.color_palette('Set1', 3)
sns.set_palette('Set1')


# # * Assignment not yet released, subject to change *

# # General instructions
# 
# 1. When instructed to implement a function, use the given function names and parameters lists; failure to do so may cause test functions to fail during grading.
# 1. When instructed to generate a plot, make sure that the plot is clear, that axes are propely labeled, and that the notebook is saved with the plot inline, so that the grader can see the plot without running the code. Make sure that you re-generate the plot if you changed the code!
# 1. Code lines with a triple comment `###` should not be removed or modified, they are used for automatic grading.
# 1. Note that there are 3 exercises and the last cell in the notebook says **end of assignment**; if you are missing anything please download the origianl file from the course website.
# 1. This exercise doesn't put much emphasis on efficieny or runtime. *But*, your code should still run within a reasonable time (a few minutes) and you should use idioms learned in class, e.g. array opreations, wherever possible to "lose your loops".
# 1. Questions regarding the exercises should be posted to the course forum at the designated group (i.e. "assignment 1"). You can post questions anonymously. You can also visit the Office Hours, but please do not email the course staff with questions about the exercise.
# 

# 
# In this exercise we'll model the spread of an infectious disease that spreads through the environment (e.g. air or water) rather than by contact with an infected person.
# 
# It would get really overwhelming if you had to ask every person at *every point in time*: "Are you sick yet? Did you get better yet?".
# It makes more sense to monitor individuals' states on a *discrete basis* rather than continuously, for example, *once a day*. 
# 
# Imagine that there's a 10% *infection rate* and a 20% *recovery rate*.
# That implies that *susceptible* people have a 10% chance to transition to the *infected* state, and *infected* people have 20% chance to transition to the *recovered* state between successive timesteps.
# 100% of those *recovered* will stay *recovered* (i.e. people who are recovered cannot become susceptible or infected again).
# In general, we will call the infection rate $\alpha$ and the recovery rate $\beta$.
# 
# This is a very simple SIR model, where $S$ stands for *susceptible*, $I$ stands for *infected*, and $R$ stands for *recovered*.
# 
# Say that you start with a population of 100 people, and only 1 person is infected (the rest are susceptible).
# That means your "initial state" is $S=99, I=1, R=0$, i.e. 99 are susceptible, 1 is infected, and 0 are recovered.

# In[5]:


S = 99 ###
R = 0 ###


# Implement a function called `step(SIR, α, β)` that given the current state `SIR=(S, I, R)` and the parameters $\alpha$ and $\beta$, randomly generates the next state.
# 
# Think: how would you implement the random draws?

# In[6]:


def step(SIR, α=0.1, β=0.2): ###
    num_of_susceptible = SIR[0]
    num_of_infected = SIR[1]
    num_of_recovered = SIR[2]   
    susceptible_array = np.random.randint(0, num_of_susceptible + 1, size=(1,num_of_susceptible))
    infected_array = np.random.randint(0, num_of_infected + 1, size=(1,num_of_infected))
    mask_infected_alpha = susceptible_array < α*num_of_susceptible
    mask_infected_beta = infected_array < β*num_of_infected
    num_of_infected_today = np.sum(mask_infected_alpha)
    num_of_recovered_today = np.sum(mask_infected_beta)
    num_of_susceptible-=num_of_infected_today
    num_of_infected+=num_of_infected_today
    num_of_infected-=num_of_recovered_today
    num_of_recovered+=num_of_recovered_today
    return np.array([num_of_susceptible, num_of_infected, num_of_recovered])    

    
step([50, 50, 0]) ### this is a stochastic function so it won't always give the same result


# Now implement a function called `simulation(SIR0, α, β, days)` that given an initial state `SIR0=(S0, I0, R0)`, parameters $\alpha$ and $\beta$, and the number of days $days$ to run the simulation, simulates the dynamics and returns a vector `SIR` in which the value at index `t, j` gives state `j` at day `t` (`j` being 0 for $S$, 1 for $I$, and for $R$). 
# 
# Note that you should call `step` from `simulation`.
# 
# Think: What is the type of the returned value? How many dimensions does it have?

# In[7]:


def simulation(SIR0, α=0.1, β=0.2, days=30): ###
    array_of_sir_vectors = np.empty((days , 3))
    current_sir = [SIR0[0],SIR0[1],SIR0[2]]
    for i in range(days):
        current_sir = step([current_sir[0],current_sir[1],current_sir[2]])
        array_of_sir_vectors[i] = current_sir
        
    return array_of_sir_vectors



print(simulation([100, 0, 0], 0.1, 0.2, 10))###


# Finally, run and plot the dynamics for 90 days, starting with 100 susceptibles, and $\alpha=0.1, \beta=0.2$.

# In[8]:


(simulation([100,0,0],days=90))
plt.plot(simulation([100,0,0],days=90))
plt.xlabel('Days')
plt.ylabel('Population size')






# # Exercise 2
# 
# In this exercise we want to estimate $T_{I}$ the time until at least 50 % of the population has already become infected (and possibly recoved) and $T_{R}$ the time until 50% of the population has recovered.
# 
# Re-implementing, when neccessary, the functions `step` and `simulation` so that you can run multiple simulations in one function call; this would require adding a new parameter for the number of replications (`reps`).
# 
# Think: what kind of an `assert` statement would be a good test of the modified functions?
# 
# Then plot the dynamics of multiple simulations.

# In[9]:


def simulation(SIR0,reps, α=0.1, β=0.2, days=30): ###
    array_of_sir_vectors = np.empty((reps,days,3))   
    simulation.array_of_ti_tr_couples = []     
    simulation.half_population_size = (SIR0[0]+SIR0[1]+SIR0[2])*0.5 + 1
    for i in range(reps):
        current_sir = np.array([SIR0[0],SIR0[1],SIR0[2]])
        TI = 0 #the number of days till we get to 50% of infected people
        TR = 0 #the number of days till we get to 50% of recovered people
        for j in range(days):
            if(current_sir[2] <= simulation.half_population_size):
                if((current_sir[1] + current_sir[2]) <= simulation.half_population_size):
                    TI +=1
                TR+=1
            current_sir = step([current_sir[0],current_sir[1],current_sir[2]])
            array_of_sir_vectors[i,j] = current_sir
        simulation.array_of_ti_tr_couples.append([TI,TR])
    return array_of_sir_vectors
        
print(simulation([100, 0, 0],4, 0.1, 0.2,10))


# In[10]:


reps = 10
result = simulation([100, 0, 0],reps, 0.1, 0.2,100)
for i in result:
    plt.plot(i, linewidth=0.3)
    plt.gca().set_prop_cycle(color=['#F8766D', '#619CFF', '#7CAE00'])
plt.xlabel('Days')
plt.ylabel('Population size')


# Re-create the previous plot but add lines for the average dynamics -- averaging across replications for each type and day.
# 
# Use at least 100 replications for this plot.

# In[11]:


reps = 100
days = 30
result = simulation([100, 0, 0],reps, 0.1, 0.2,30)
result_avg = np.average(result,axis=0)
for i in result:
    plt.plot(i, linewidth=0.3)
    plt.gca().set_prop_cycle(color=['#F8766D', '#619CFF', '#7CAE00'])
    plt.plot(result_avg, "-k")
plt.xlabel('Days')
plt.ylabel('Population size')


# Now print the $T_{I}$ the first day at which at least 50% of the *average* population was already infected and $T_{R}$ the first day at which at least 50% of the *average* population is recovered.
# 
# Remember: the calculation should not assume a specific population size (i.e. 100 people).

# In[12]:


T_I = 0
T_R = 0
for i in range(days):
    if result_avg[i][2] < simulation.half_population_size:
        T_I += 1
        if (result_avg[i][1] + result_avg[i][2]) < simulation.half_population_size:
            T_R += 1
print('T_I = ', T_I) ###
print('T_R = ', T_R) ###


# Repeast the previous plot (dynamics + average) and add vertical lines for $T_{I}$ and $T_{R}$ (using `plt.axvline`).
# Zoom in to the first 30 days and also add a horizontal line at 50% (using `plt.axhline`).

# In[13]:


T_I = 0
T_R = 0
reps = 100
days = 40
result = simulation([100, 0, 0],reps, 0.1, 0.2,days)
result_avg = np.average(result,axis=0)
for i in result:
    plt.plot(i, linewidth=0.3)
    plt.gca().set_prop_cycle(color=['#F8766D', '#619CFF', '#7CAE00'])
plt.plot(result_avg, "-k")

for i in range(days):
    if result_avg[i][2] < simulation.half_population_size:
        T_I += 1
        if (result_avg[i][1] + result_avg[i][2]) < simulation.half_population_size:
            T_R += 1

plt.axvline(T_I,0,1,linestyle = '--',color = 'k')
plt.axvline(T_R,0,1,linestyle = '--',color = 'k')
plt.axhline(simulation.half_population_size, linestyle = '-', color = 'k')
plt.axis([0,30,0,simulation.half_population_size*2])
plt.xlabel('Days')
plt.ylabel('Population size')
print('T_I = ', T_I) ###
print('T_R = ', T_R) ###






# # Exercise 3
# 
# Now we change our focus to a disease that is transmitted by contact.
# 
# Therefore, the probability for a susceptible individual to become infected depends on the frequency of infected individuals, $I/N$ where $N=S+I+R$, and the infection rate is therefore $\alpha I/N$ instead of just $\alpha$.
# The recovery rate does not change.
# 
# Implement `step` and/or `simulation` accordingly to find $T_{I}$ and $T_{R}$ for this disease.
# 
# You will also find that the $\alpha$ and $\beta$ parameters we used before do not allow the disease to spread (which is great!); find some parameters that do allow the disease to spread within a reasonable time scale (i.e. $T_R<365$, that is, less than a year).
# Because you will use your own $\alpha$ and $\beta$ parameters your results ($T_I$, $T_R$ and plot) will differ from below.

# In[14]:


def step(SIR, α = 0.3, β = 0.02): ###
    num_of_susceptible = SIR[0]
    num_of_infected = SIR[1]
    num_of_recovered = SIR[2] 
    total_population = num_of_susceptible + num_of_infected + num_of_recovered #total population is always constant
    rate_of_infection = ((num_of_infected/total_population)*α)
    susceptible_array = np.random.random(size=num_of_susceptible)
    infected_array = np.random.random(size=num_of_infected)
    mask_infected_alpha = susceptible_array < (rate_of_infection)
    mask_infected_beta = infected_array < β
    num_of_infected_today = np.sum(mask_infected_alpha)
    num_of_recovered_today = np.sum(mask_infected_beta)
    num_of_susceptible-=num_of_infected_today
    num_of_infected+=num_of_infected_today
    num_of_infected-=num_of_recovered_today
    num_of_recovered+=num_of_recovered_today
    return np.array([num_of_susceptible, num_of_infected, num_of_recovered])    

    
step([99, 1, 0]) ### this is a stochastic function so it won't always give the same result


# In[15]:


α = 0.3 # please insert a value here
β = 0.02 # please insert a value here
print('α =', α) ###
print('β =', β) ###


# In[17]:


T_I, T_R = 0,0
reps = 80
days = 180
final_result = simulation([99, 1, 0],reps, α, β,days)
result_avg = np.average(final_result,axis=0)
for i in range(days):
    if result_avg[i][2] < simulation.half_population_size:
        T_I += 1
        if (result_avg[i][1] + result_avg[i][2]) < simulation.half_population_size:
            T_R += 1
print('T_I = ', T_I) ###
print('T_R = ', T_R) ###


# In[18]:


for i in final_result:
    plt.plot(i, linewidth=0.3)
    plt.gca().set_prop_cycle(color=['#F8766D', '#619CFF', '#7CAE00'])
plt.axvline(T_I,0,1,linestyle = '--',color = 'k')
plt.axvline(T_R,0,1,linestyle = '--',color = 'k')
plt.axhline(simulation.half_population_size, linestyle = '-', color = 'k')
plt.xlabel('Days')
plt.ylabel('Population size')
plt.plot(result_avg, "-k")


# **End of assignment**
