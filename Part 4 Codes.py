# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 01:02:48 2018

@author: Johnny
"""

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt


#For stock price simulation
def simulate_BS_Stock_Movement(S0, sigma, r, T, paths, steps, K):
    
    error = []
    # Black scholes stock movement 
    deltaT=T/steps
    t=np.linspace(0,T,steps+1)
    X=np.c_[np.zeros((paths,1)),
            np.random.randn(paths,steps)]
    St = S0*np.exp((r-(sigma**2)/2)*(t) + (sigma * np.cumsum(np.sqrt(deltaT) * X, axis=1)))


    #For every BS path
    for BS_path in St:

        cash = []
        prev_Delta = 0
        
        c0 = BlackScholesCall(BS_path[0],K,r,sigma,T)*np.exp(r*(T))
        
        #for every time step in each path
        for idx in range(0, len(BS_path)):
            
            Delta= Black_Scholes_Greeks_Call(BS_path[idx],K,r,sigma,(T-t[idx]))
            
            if idx != 0:
                #dynamic delta hedging (buy delta shares to hedge) with interest rate
                cash.append((-Delta--prev_Delta) * BS_path[idx]*np.exp(r*(T-t[idx])))
            else:
                cash.append((-Delta)*BS_path[idx]*np.exp(r*(T-t[idx])))  
            prev_Delta = Delta
        
        if BS_path[steps] < 100:
             error.append(np.sum(cash) + c0)
        else:
            error.append(np.sum(cash) + K + c0)

    return t,St.T,error

#BS Call
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

##################

plt.figure()
t,x,error_21 = simulate_BS_Stock_Movement(S0, sigma, r, T, paths, 21, K)
plt.title('Black Scholes underlying price paths at N=21')
plt.plot(t,x)
plt.xlabel('Time (t)')
plt.ylabel('Underlying Price ($)')
plt.savefig('BS_price_21.jpg', format='jpg', dpi=400)
plt.show()

plt.figure()
plt.hist(error_21, weights=(np.zeros_like(np.array(error_21))+1/ np.array(error_21).size)*100,bins=20)
plt.title('Hedging error with N = 21')
plt.xlabel('Final P&L ($)')
plt.ylabel('Frequency (%)')
plt.grid()
plt.savefig('error_21.jpg', format='jpg', dpi=400)
plt.show()
print('Mean of N=21 : %0.3f'%np.mean(error_21))
print('Std of N=21 : %0.3f'%np.std(error_21))


#################

t,x,error_84  = simulate_BS_Stock_Movement(S0, sigma, r, T, paths, 84, K)
plt.title('Black Scholes underlying price paths at N=84')
plt.plot(t,x)
plt.xlabel('Time (t)')
plt.ylabel('Underlying Price ($)')
plt.savefig('BS_price_84.jpg', format='jpg', dpi=400)
plt.show()

plt.hist(error_84, weights=(np.zeros_like(np.array(error_84))+1/ np.array(error_84).size)*100,bins=20)
plt.title('Hedging error with N = 84')
plt.xlabel('Final P&L ($)')
plt.ylabel('Frequency (%)')
plt.grid()
plt.savefig('error_84.jpg', format='jpg', dpi=400)
plt.show()
print('Mean of N=84 : %0.3f'%np.mean(error_84))
print('Std of N=84 : %0.3f'%np.std(error_84))
##################
