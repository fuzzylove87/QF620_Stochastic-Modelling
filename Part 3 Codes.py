import pandas as pd
import numpy as np
import scipy.stats as ss
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pylab as plt
import datetime as dt
from scipy import interpolate
from math import exp,log
from scipy.integrate import quad

#1.Part 1 Option pricers
def BS_Call(S,K,T,r,sigma):
    d1 = (np.log(S/K)+(r+(sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = (np.log(S/K)+(r-(sigma**2)/2)*T)/(sigma*np.sqrt(T))
    return S*ss.norm.cdf(d1,0,1)-K*np.exp(-r*T)*ss.norm.cdf(d2,0,1)

def BS_Put(S,K,T,r,sigma):
    d1 = (np.log(S/K)+(r+(sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = (np.log(S/K)+(r-(sigma**2)/2)*T)/(sigma*np.sqrt(T))
    return K*np.exp(-r*T)*ss.norm.cdf(-d2,0,1)-S*ss.norm.cdf(-d1,0,1)

def Bach_Call(S,K,T,r,sigma):
    x_Star = (K-S)/(S*sigma*np.sqrt(T))
    return np.exp(-r*T)*((S-K)*ss.norm.cdf(-x_Star,0,1)+
                         S*sigma*np.sqrt(T)*ss.norm.pdf(-x_Star,0,1))

def Bach_Put(S,K,T,r,sigma):
    x_Star = (K-S)/(S*sigma*np.sqrt(T))
    return np.exp(-r*T)*((K-S)*ss.norm.cdf(x_Star,0,1)+
                         S*sigma*np.sqrt(T)*ss.norm.pdf(-x_Star,0,1))

def B76_LogN_Call(S,K,T,r,sigma):
    F=S*np.exp(r*T)
    d1 = (np.log(F/K)+(sigma**2)/2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(F/K)-(sigma**2)/2*T)/(sigma*np.sqrt(T))
    return np.exp(-r*T)*(F*ss.norm.cdf(d1,0,1)-K*ss.norm.cdf(d2,0,1))


def B76_LogN_Put(S,K,T,r,sigma):
    F=S*np.exp(r*T)
    d1 = (np.log(F/K)+(sigma**2)/2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(F/K)-(sigma**2)/2*T)/(sigma*np.sqrt(T))
    return np.exp(-r*T)*(K*ss.norm.cdf(-d2,0,1)-F*ss.norm.cdf(-d1,0,1))

#2.Import data and SABR

#Default value Set
today=dt.date(2013,8,30)
expiry=dt.date(2015,1,17)
D=(expiry-today).days
T=D/365.0
rate_df = pd.read_csv('discount.csv')
interp = interpolate.interp1d(rate_df.iloc[:,0],rate_df.iloc[:,1])
#print('interp(D)',interp(D)) # Percent terms of D value in interpolate object
S = 846.9
r = interp(D)/100
F = S*np.exp(r*T)

sigma = (0.25425618293224583 + 0.26229211463233354)/2
print('sigma',sigma)

#improt data

def fetch():
    
    googcall = pd.read_csv('goog_call.csv')
    googcall['call'] = (googcall['best_bid']+googcall['best_offer'])/2
#    googcall.drop(['date','expiry','best_bid','best_offer'],axis=1,inplace=True)
    
    googput = pd.read_csv('goog_put.csv')
    googput['put'] = (googput['best_bid']+googput['best_offer'])/2
#    googput.drop(['date','expiry','best_bid','best_offer'],axis=1,inplace=True)
    
    global goog 
    goog = pd.concat([googcall['strike'],googcall['call'],googput['put']], axis=1)
    return goog

fetch()
goog['callonly']= goog.apply(
    lambda row: row['call'] if row['strike']>F else np.nan,
    axis=1
)
goog['putonly']= goog.apply(
    lambda row: row['put'] if row['strike']<F else np.nan,
    axis=1
)


strikes = goog['strike']
strikes_c = [i for i in goog['strike'] if i>F]
strikes_p = [i for i in goog['strike'] if i<F]
call_price = [i for i in goog['callonly'] if i>0]
put_price = [i for i in goog['putonly'] if i>0]


beta=0.8
alpha=0.992036
rho=-0.285001
nu=0.352716


def SABR(F, K, T, alpha, beta, rho, nu):
    X = K
    if F == K:
        numer1 = (((1 - beta)**2)/24)*alpha*alpha/(F**(2 - 2*beta))
        numer2 = 0.25*rho*beta*nu*alpha/(F**(1 - beta))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        VolAtm = alpha*(1 + (numer1 + numer2 + numer3)*T)/(F**(1-beta))
        sabrsigma = VolAtm
    else:
        z = (nu/alpha)*((F*X)**(0.5*(1-beta)))*np.log(F/X)
        zhi = np.log((((1 - 2*rho*z + z*z)**0.5) + z - rho)/(1 - rho))
        numer1 = (((1 - beta)**2)/24)*((alpha*alpha)/((F*X)**(1 - beta)))
        numer2 = 0.25*rho*beta*nu*alpha/((F*X)**((1 - beta)/2))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        numer = alpha*(1 + (numer1 + numer2 + numer3)*T)*z
        denom1 = ((1 - beta)**2/24)*(np.log(F/X))**2
        denom2 = (((1 - beta)**4)/1920)*((np.log(F/X))**4)
        denom = ((F*X)**((1 - beta)/2))*(1 + denom1 + denom2)*zhi
        sabrsigma = numer/denom

    return sabrsigma


sabr1=[]
for i in strikes:
    sabr1.append(SABR(F,i,T, alpha,beta,rho,nu))
sabr1 = pd.DataFrame(sabr1, columns=['SABRsigma'])



#3.Part 3 Static Replication
#1.Payoff function
#1.1 B-S

def BS_Derive(S,T,r,sigma):
    num1 = (S**3)*exp(3*T*(r+sigma**2))
    num2 = 2.5*(log(S)+T*(r-(sigma**2)/2))
    num3 = 10.0
    value=exp(-r*T)*(num1+num2+num3)
    return value
print('1.BS model value',BS_Derive(S,T,r,sigma)) #sigma is sabrsigma at F=K


#1.2 Bachelier model

def Bach_Derive(S,T,r,sigma):
    num1=(S**3)*(1+3*T*(sigma**2))
    func=lambda x:log(1+sigma*np.sqrt(T)*x)*exp(-0.5*(x**2))
    bound=-1/(sigma*np.sqrt(T))
    num2=log(S)+quad(func,bound,np.inf)[0]
    num3=10.0
    value=exp(-r*T)*(num1+num2+num3)
    return value
print('2.Bachelier model value',Bach_Derive(S,T,r,sigma))

#1.3.Static-replication of European payoff

def H_0(S):
    return S**3+log(S)+10.0
def H_1(S):
    return 3*(S**2)+2.5/S
def H_2(S):
    return 6*S-2.5/(S**2)

def Static_replication(F,PK,CK,T,r):
    func1 = lambda x:PK(S,x,T,r,SABR(F,x,T,alpha,beta,rho,nu))*H_2(x)
    func2 = lambda x:CK(S,x,T,r,SABR(F,x,T,alpha,beta,rho,nu))*H_2(x)
    num1=exp(-r*T)*H_0(F)
    num2=quad(func1,0,F)
    num3=quad(func2,F,np.inf)
    return num1+num2[0]+num3[0]

print('3.Static-replication value',Static_replication(F,B76_LogN_Put,B76_LogN_Call,T,r))


#2.Model-Free integrated variance
#2.1 B-S model
print('1.B-S model sigma',sigma)

#2.2 Bachelier Model

def MFiv_B(PK,CK,sigma):
    func1=lambda x:PK(S,x,T,r,sigma)/(x**2)
    func2=lambda x:CK(S,x,T,r,sigma)/(x**2)
    num1=quad(func1,0,F)
    num2=quad(func2,F,np.inf)
    return 2*exp(r*T)*(num1[0]+num2[0])/T


print('2.Bachelier model sigma',MFiv_B(Bach_Put,Bach_Call,sigma))


#2.3 Static-replication of European payoff
def MFiv_Static(PK,CK):
    func1=lambda x:PK(S,x,T,r,SABR(F,x,T,alpha,beta,rho,nu))/(x**2)
    func2=lambda x:CK(S,x,T,r,SABR(F,x,T,alpha,beta,rho,nu))/(x**2)
    num1=quad(func1,0,F)
    num2=quad(func2,F,np.inf)
    return 2*exp(r*T)*(num1[0]+num2[0])/T


print('3.Static-replication sigma',MFiv_Static(B76_LogN_Put,B76_LogN_Call))
