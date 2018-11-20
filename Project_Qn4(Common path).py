# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 01:02:48 2018

@author: Johnny
"""

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

#For stock price simulation
def simulate_BS_Stock_Movement(S0, sigma, r, T, paths, steps, hedge_freq, K):
    
    # Using one comon black scholes stock movement with steps = 84 
    deltaT=T/steps
    t=np.linspace(0,T,steps+1)
    X=np.c_[np.zeros((paths,1)),
            np.random.randn(paths,steps)]
    St = S0*np.exp((r-(sigma**2)/2)*(t) + (sigma * np.cumsum(np.sqrt(deltaT) * X, axis=1)))

    #To call the BlackScholesCall at every timestep
    hedge_errors = []
    for hf in hedge_freq:

        error = []
        for BS_path in St:
            
            cash = []
            prev_Delta = 0
            
            c0 = BlackScholesCall(BS_path[0],K,r,sigma,T)*np.exp(r*(T))
        
            # Slicing range by 1,4 depending on frequency
            for idx in range(0, len(BS_path), hf):
                
                Delta= Black_Scholes_Greeks_Call(BS_path[idx],K,r,sigma,(T-t[idx]))
                
                if idx != 0:
                    #dynamic delta hedging (buy delta shares to hedge) with interest rate
                    cash.append((-Delta--prev_Delta) * BS_path[idx]*np.exp(r*(T-t[idx])))
                else:
                    cash.append((-Delta)*BS_path[idx]*np.exp(r*(T-t[idx])))  
                prev_Delta = Delta
            print(np.sum(cash),BS_path[21])
            
            if BS_path[steps] < 100:
                 error.append(np.sum(cash) + c0)
            else:
                error.append(np.sum(cash) + K + c0)
        hedge_errors.append(error)
    return t,St.T,hedge_errors

#BS call price (aka hedge costs)
def BlackScholesCall(S, K, r, sigma, T):
    #T to be fraction of years
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*stats.norm.cdf(d1) - K*np.exp(-r*T)*stats.norm.cdf(d2)

def Black_Scholes_Greeks_Call(S, K, r, v, T):
    T_sqrt = np.sqrt(T)
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*T_sqrt)
    Delta = stats.norm.cdf(d1)
    return Delta

S0=100
sigma=0.2
r=0.05
T=1/12
paths=50000 #to be set 50000
K=S0 # ATM strike
steps = 84
hedge_freq = [1,4]


#################
plt.figure()
t,x,hedge_errors = simulate_BS_Stock_Movement(S0, sigma, r, T, paths, steps, hedge_freq, K)
plt.title('Black Scholes movement with N = 84')
plt.plot(t,x)
plt.show()

plt.hist(hedge_errors[0], weights=(np.zeros_like(np.array(hedge_errors[0]))+1/ np.array(hedge_errors[0]).size)*100,bins=10)
plt.title('Hedging error with N = 21')
plt.xlabel('Final P&L ($)')
plt.ylabel('Frequency (%)')
plt.show()
print(np.mean(hedge_errors[0]),np.std(hedge_errors[0]))
#################



##################
#plt.figure()
#t,x,error_84 = simulate_BS_Stock_Movement(S0, sigma, r, T, paths, steps, K)
#plt.title('Black Scholes movement with N = 84')
#plt.plot(t,x)
#plt.show()

plt.hist(hedge_errors[1], weights=(np.zeros_like(np.array(hedge_errors[1]))+1/ np.array(hedge_errors[1]).size)*100,bins=10)
plt.title('Hedging error with N = 84')
plt.xlabel('Final P&L ($)')
plt.ylabel('Frequency (%)')
plt.show()
print(np.mean(hedge_errors[1]),np.std(hedge_errors[1]))
##################
