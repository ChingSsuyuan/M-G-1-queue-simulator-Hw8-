"""
Simulation of an M|G|1 queue with one class
Uses simpy to dynamically generate events 

The simulator uses the Gamma distribution for service times, 
with a hardcoded exception for the deterministic distribution for second moment
1/MU^2. This has advantage of seeing how changing the service distribution 
changes the results. In addition, Gamma distribution with SHAPE = 1 (or K=2) 
corresponds to the Exponential distribution

Authors: Jonathan Chamberlain and David Starobinski
Modified by Siyuan Jing siyuan16@bu.edu
"""



# import required packages - 
# numpy, scipy, and simpy required to be installed if not present

import math
import numpy as np
import scipy as sp
import scipy.stats as stats
import simpy
import collections
import matplotlib.pyplot as plt


'''
Define Simulation Global Parameters
'''

LAM = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]# Arrival rates of customers
NUMLAM = len(LAM)

MU = 1 # Service rate of customers; defined as 1 over first moment of service

LAM1=[lam/2 for lam in LAM]
LAM2=[lam/2 for lam in LAM]
K = 3 # Service Distribution; second moment of service is K over MU^2

if K < 1:
    print('K must be at least 1')
    exit()

# RHO = np.zeros(NUMLAM) # load for each run
RHO1=np.array(LAM1/2)/MU
RHO2=np.array(LAM2/2)/MU
RHO=RHO1+RHO2
for l in range(NUMLAM):
    RHO[l] = LAM[l]/MU
    if RHO[l] >=1:
        print('Unstable system specified. Lambda should be less than Mu.')
        exit()
        

FRAC = 0.1 # fraction of time to wait for before collecting statistics

ITERATIONS = 30 # number of independent simulations

ALPHA = 0.05 # confidence interval is 100*(1-alpha) percent

PKT_NUM = 10**4 #average number of arrivals in each simulation

# define parameters of Gamma distribution; Numpy uses shape/scale definition
if K > 1:
    SHAPE = 1/(K-1) # Shape of Gamma Distribution
    SCALE = (K-1)/MU # Scale of Gamma Distribution


'''
Create the provider to serve the customers

env - SimPy Environment
arrival - the customer's arrival time to the queue
serv_time - the length of service of the job; randomly generated on arrival
server - tuple featuring server resource and delay statistic collector
t_start - time to begin collection of statistics
'''

def provider(env,arrival,serv_time,t_start,server):
    # yield until the server is available
    with server.processor.request(priority=priority) as MyTurn:
        yield MyTurn

        # customer has acquired the server, run job for specified service time
        yield env.timeout(serv_time)

        # Record total system time, if beyond the initial transient time
        if (env.now > t_start):
            server.delay[0] += env.now-arrival
            server.n[0] += 1


'''
Create stream of customers until SIM_TIME reached

env - the SimPy Environment
server - tuple featuring server resource and delay statistic collector
rate - arrival rate passed from loop
t_start - time to begin collection of statistics
'''

def arrivals(env, server, rate, t_start):
    while True:

        yield env.timeout(np.random.exponential(1/rate)) # Poisson arrivals; 

        arrival = env.now # mark arrival time
        
        if K == 1: 
            serv_time = 1/MU # Special case for Deterministic system
        else:
            serv_time = np.random.gamma(SHAPE,SCALE)

        # Have server process customer arrival
        env.process(provider(env,arrival,serv_time,t_start,server))

'''
Define supporting structures
'''

# define server tuple to pass into arrivals, provider methods
Server = collections.namedtuple('Server','processor,delay,n') 

# Mean delay in each iteration
Mean_Delay = np.zeros((ITERATIONS,NUMLAM)) 

'''
Main Simulator Loop
'''
for l in range(NUMLAM):
    for itr in range(ITERATIONS):
        print('Lambda %.3f, Iteration # %d' %(LAM[l],itr))

        env = simpy.Environment() # establish SimPy environment
        # M|G|1 server, could simulate arbitrary M|G|n by updating capacity
        processor=simpy.PriorityResource(env,capacity=1)

        delay = np.zeros(1)
        n = np.zeros(1)
        rate = LAM[l]

        # Length of simulation to generate PKT_NUM arrivals on average
        sim_time = PKT_NUM/rate 

        t_start = FRAC*sim_time # Start collecting statistics after that time

        server = Server(processor,delay,n) #define Server collections

        #start simulation
        env.process(arrivals(env,server,rate,t_start)) 
        env.run(until=sim_time)

        # Record average delay
        Mean_Delay[itr,l] = delay[0]/n[0]
    
        

'''
Compute Statistics     
'''
print('Mean Delay Vector \r\n', Mean_Delay)
Sample_Delay = np.mean(Mean_Delay,axis=0) # Sample Mean of the Delays

# compute confidence intervals
print('Statistical Results')
CI = stats.sem(Mean_Delay, axis=0)*stats.norm.ppf(1-ALPHA/2)


for l in range(NUMLAM):
    print('At arrival rate %f:' %(LAM[l]))
    print('Sample Delay is %.3f with confidence interval %.3f.' %(Sample_Delay[l],CI[l]))
'''
Plot Statistical Results against Analytical Expected Values
'''

NPAnalytical_Delay = np.zeros(NUMLAM) # Expected Delay

for l in range(NUMLAM):  #PK Formula
    NPAnalytical_Delay[l] = (K*RHO[l])/(2*MU*(1-RHO[l])) + 1/MU  

# Plot of Expected Delays    
plt.plot(LAM,NPAnalytical_Delay, label='Analytical')

# Plot of Simulated Delays
plt.errorbar(LAM, Sample_Delay, yerr=CI, fmt='x', label='Simulated') 
plt.title('Comparison of Analysis to Simulation (K=%d, MU=%.3f)' %(K, MU))
plt.xlabel('Lambda')
plt.ylabel('Mean System Time (Delay)')
plt.legend()
plt.show()  
''' Save the figure in PDF format - Need to comment out plt.show
plt.savefig('MG1-bars.pdf')
'''
#plt.savefig('MG1-bars.pdf')
