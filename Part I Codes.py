# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:59:24 2018

@author: Woon Tian Yong
"""

import numpy as np
import pandas as pd
import scipy.stats as ss

# Standalone VANILLA Option Functions

def BS_Call(S,K,T,r,sigma):
    d1 = (np.log(S/K)+(r+(sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = (np.log(S/K)+(r-(sigma**2)/2)*T)/(sigma*np.sqrt(T))
    return S*ss.norm.cdf(d1)-K*np.exp(-r*T)*ss.norm.cdf(d2)

def BS_Put(S,K,T,r,sigma):
    d1 = (np.log(S/K)+(r+(sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = (np.log(S/K)+(r-(sigma**2)/2)*T)/(sigma*np.sqrt(T))
    return K*np.exp(-r*T)*ss.norm.cdf(-d2)-S*ss.norm.cdf(-d1)

def B76_LogN_Call(S,K,T,r,sigma):
    F = S*np.exp(r*T)
    x_star = (np.log(K/F)+(sigma**2)*T/2)/(sigma*np.sqrt(T))
    return np.exp(-r*T)*(F*ss.norm.cdf(-x_star+sigma*np.sqrt(T)) \
            -K*ss.norm.cdf(-x_star))

def B76_LogN_Put(S,K,T,r,sigma):
    F = S*np.exp(r*T)
    x_star = (np.log(K/F)+(sigma**2)*T/2)/(sigma*np.sqrt(T))
    return np.exp(-r*T)*(K*ss.norm.cdf(x_star) \
            -F*ss.norm.cdf(x_star-sigma*np.sqrt(T)))

def Bach_Call(S,K,T,sigma,r=0):
    x_star = (K-S)/(S*sigma*np.sqrt(T))
    return np.exp(-r*T)*((S-K)*ss.norm.cdf(-x_star)+
                         S*sigma*np.sqrt(T)*ss.norm.pdf(-x_star))

def Bach_Put(S,K,T,sigma,r=0):
    x_star = (K-S)/(S*sigma*np.sqrt(T))
    return np.exp(-r*T)*((K-S)*ss.norm.cdf(x_star)+
                         S*sigma*np.sqrt(T)*ss.norm.pdf(x_star))

def B76_N_Call(S,K,T,r,sigma):
    F = S*np.exp(r*T)
    x_star = (K-F)/(F*sigma*np.sqrt(T))
    return np.exp(-r*T)*((F-K)*ss.norm.cdf(-x_star)+
            F*sigma*np.sqrt(T)*ss.norm.pdf(-x_star))

def B76_N_Put(S,K,T,r,sigma):
    F = S*np.exp(r*T)
    x_star = (K-F)/(F*sigma*np.sqrt(T))
    return np.exp(-r*T)*((K-F)*ss.norm.cdf(x_star)+
            F*sigma*np.sqrt(T)*ss.norm.pdf(x_star))    

def DD_Call(S,K,T,r,sigma,Beta):
    if Beta==0:
        F=(S*np.exp(r*T))
        x_star = (K-F)/(F*sigma*np.sqrt(T))
        return np.exp(-r*T)*((F-K)*ss.norm.cdf(-x_star)+
                F*sigma*np.sqrt(T)*ss.norm.pdf(-x_star))
    else:
        F=(S*np.exp(r*T))/Beta
        K=K+((1-Beta))*F
        sigma=Beta*sigma
        x_star = (np.log(K/F)+(sigma**2)*T/2)/(sigma*np.sqrt(T))
        return np.exp(-r*T)*(F*ss.norm.cdf(-x_star+sigma*np.sqrt(T)) \
                -K*ss.norm.cdf(-x_star))

def DD_Put(S,K,T,r,sigma,Beta):
    if Beta==0:
        F=(S*np.exp(r*T))
        x_star = (K-F)/(F*sigma*np.sqrt(T))
        return np.exp(-r*T)*((K-F)*ss.norm.cdf(x_star)+
                F*sigma*np.sqrt(T)*ss.norm.pdf(x_star))
    else:
        F=(S*np.exp(r*T))/Beta
        K=K+((1-Beta))*F
        sigma=Beta*sigma
        x_star = (np.log(K/F)+(sigma**2)*T/2)/(sigma*np.sqrt(T))
        return np.exp(-r*T)*(K*ss.norm.cdf(x_star) \
                -F*ss.norm.cdf(x_star-sigma*np.sqrt(T))) 

# Standalone DIGITAL CASH OR NOTHING Option Functions    

def BS_CoN_Call(S,K,T,r,sigma):
    x_star = (np.log(K/S)-(r-(sigma**2)/2)*T)/(sigma*np.sqrt(T))
    return np.exp(-r*T)*ss.norm.cdf(-x_star)
    
def BS_CoN_Put(S,K,T,r,sigma):
    x_star = (np.log(K/S)-(r-(sigma**2)/2)*T)/(sigma*np.sqrt(T))
    return np.exp(-r*T)*ss.norm.cdf(x_star)

def B76_LogN_CoN_Call(S,K,T,r,sigma):
    F = S*np.exp(r*T)
    x_star = (np.log(K/F)+(sigma**2)*T/2)/(sigma*np.sqrt(T))
    return np.exp(-r*T)*ss.norm.cdf(-x_star)
    
def B76_LogN_CoN_Put(S,K,T,r,sigma):
    F = S*np.exp(r*T)
    x_star = (np.log(K/F)+(sigma**2)*T/2)/(sigma*np.sqrt(T))   
    return np.exp(-r*T)*ss.norm.cdf(x_star)

def Bach_CoN_Call(S,K,T,sigma,r=0):
    x_star = (K-S)/(S*sigma*np.sqrt(T))
    return np.exp(-r*T)*ss.norm.cdf(-x_star)
    
def Bach_CoN_Put(S,K,T,sigma,r=0):
    x_star = (K-S)/(S*sigma*np.sqrt(T))  
    return np.exp(-r*T)*ss.norm.cdf(x_star)

def B76_N_CoN_Call(S,K,T,r,sigma):
    F = S*np.exp(r*T)
    x_star = (K-F)/(F*sigma*np.sqrt(T))
    return np.exp(-r*T)*ss.norm.cdf(-x_star)

def B76_N_CoN_Put(S,K,T,r,sigma):
    F = S*np.exp(r*T)
    x_star = (K-F)/(F*sigma*np.sqrt(T)) 
    return np.exp(-r*T)*ss.norm.cdf(x_star)

def DD_CoN_Call(S,K,T,r,sigma,Beta):
    if Beta==0:   
        F = S*np.exp(r*T)
        x_star = (K-F)/(F*sigma*np.sqrt(T))
        return np.exp(-r*T)*ss.norm.cdf(-x_star)
    else: 
        F=(S*np.exp(r*T))/Beta
        K=K+((1-Beta))*F
        sigma=Beta*sigma
        x_star = (np.log(K/F)+(sigma**2)*T/2)/(sigma*np.sqrt(T))
        return np.exp(-r*T)*ss.norm.cdf(-x_star)
    
def DD_CoN_Put(S,K,T,r,sigma,Beta):
    if Beta==0:   
        F = S*np.exp(r*T)
        x_star = (K-F)/(F*sigma*np.sqrt(T))
        return np.exp(-r*T)*ss.norm.cdf(x_star)
    else: 
        F=(S*np.exp(r*T))/Beta
        K=K+((1-Beta))*F
        sigma=Beta*sigma
        x_star = (np.log(K/F)+(sigma**2)*T/2)/(sigma*np.sqrt(T)) 
        return np.exp(-r*T)*ss.norm.cdf(x_star)
    
# Standalone DIGITAL ASSET OR NOTHING Option Functions  
    
def BS_AoN_Call(S,K,T,r,sigma):
    x_star = (np.log(K/S)-(r-(sigma**2)/2)*T)/(sigma*np.sqrt(T))
    return S*ss.norm.cdf(-x_star+sigma*np.sqrt(T))
    
def BS_AoN_Put(S,K,T,r,sigma):
    x_star = (np.log(K/S)-(r-(sigma**2)/2)*T)/(sigma*np.sqrt(T))
    return S*ss.norm.cdf(x_star-sigma*np.sqrt(T))
    
def B76_LogN_AoN_Call(S,K,T,r,sigma):
    F = S*np.exp(r*T)
    x_star = (np.log(K/F)+(sigma**2)*T/2)/(sigma*np.sqrt(T)) 
    return np.exp(-r*T)*F*ss.norm.cdf(-x_star+sigma*np.sqrt(T))
    
def B76_LogN_AoN_Put(S,K,T,r,sigma):
    F = S*np.exp(r*T)
    x_star = (np.log(K/F)+(sigma**2)*T/2)/(sigma*np.sqrt(T))
    return np.exp(-r*T)*F*ss.norm.cdf(x_star-sigma*np.sqrt(T))
    
def Bach_AoN_Call(S,K,T,sigma,r=0):
    x_star = (K-S)/(S*sigma*np.sqrt(T)) 
    return np.exp(-r*T)*S*(ss.norm.cdf(-x_star)
            +sigma*np.sqrt(T)*ss.norm.pdf(-x_star))
    
def Bach_AoN_Put(S,K,T,sigma,r=0):
    x_star = (K-S)/(S*sigma*np.sqrt(T)) 
    return np.exp(-r*T)*S*(ss.norm.cdf(x_star)
            -sigma*np.sqrt(T)*ss.norm.pdf(x_star))
    
def B76_N_AoN_Call(S,K,T,r,sigma):
    F = S*np.exp(r*T)
    x_star = (K-F)/(F*sigma*np.sqrt(T)) 
    return np.exp(-r*T)*F*(ss.norm.cdf(-x_star)
            +sigma*np.sqrt(T)*ss.norm.pdf(-x_star))
    
def B76_N_AoN_Put(S,K,T,r,sigma):
    F = S*np.exp(r*T)
    x_star = (K-F)/(F*sigma*np.sqrt(T))  
    return np.exp(-r*T)*F*(ss.norm.cdf(x_star)
            -sigma*np.sqrt(T)*ss.norm.pdf(x_star))
        
def DD_AoN_Call(S,K,T,r,sigma,Beta):
    if Beta==0:   
        # can call B76_N_AoN_Call
        F = S*np.exp(r*T)
        x_star = (K-F)/(F*sigma*np.sqrt(T)) 
        return np.exp(-r*T)*F*(ss.norm.cdf(-x_star)
                   +sigma*np.sqrt(T)*ss.norm.pdf(-x_star))
    else: 
        F=(S*np.exp(r*T))/Beta
        K=K+((1-Beta))*F
        sigma=Beta*sigma
        x_star = (np.log(K/F)+(sigma**2)*T/2)/(sigma*np.sqrt(T)) 
        return np.exp(-r*T)*F*ss.norm.cdf(-x_star+sigma*np.sqrt(T))
        
def DD_AoN_Put(S,K,T,r,sigma,Beta):
    if Beta==0:   
        F = S*np.exp(r*T)
        x_star = (K-F)/(F*sigma*np.sqrt(T))
        return np.exp(-r*T)*F*(ss.norm.cdf(x_star)
               -sigma*np.sqrt(T)*ss.norm.pdf(x_star))
    else: 
        F=(S*np.exp(r*T))/Beta
        K=K+((1-Beta))*F
        sigma=Beta*sigma
        x_star = (np.log(K/F)+(sigma**2)*T/2)/(sigma*np.sqrt(T))  
        return np.exp(-r*T)*F*ss.norm.cdf(x_star-sigma*np.sqrt(T))  