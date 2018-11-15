# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:31:30 2018

@author: Patrick
"""

#Assignment 10
#number 1
import pandas as pd
from plotnine import *
from scipy.optimize import minimize
from scipy.stats import norm
import numpy as np
from scipy import stats

#load data
data=pd.read_csv('data.txt',header=0,sep=',')

#Create two likelihood functions
def hump(p,obs):
    B0=p[0]
    B1=p[1]
    B2=p[2]
    sigma=p[3]
    
    pred=B0+B1*data.x+B2*data.x**2
    
    nll=-1*norm(pred,sigma).logpdf(data.y).sum()
    return nll

def linear(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    
    pred=B0+B1*data.x
    
    nll=-1*norm(pred,sigma).logpdf(data.y).sum()
    return nll

#estimate parameterw
humpGuess=np.array([1,1,1,1])
linearGuess=np.array([1,1,1])

fithump=minimize(hump,humpGuess,method='Nelder-Mead',args='data')
fitlinear=minimize(linear,linearGuess,method='Nelder-Mead', args='data')

#run a likelihood ratio test
teststat=2*(fitlinear.fun-fithump.fun)
df=len(fithump.x)-len(fitlinear.x)

p=1-stats.chi2.cdf(teststat,df)

#analyze the result and print the analysis
if p<=0.05:
    print ("Quadratic approximation is more accurate")
elif p>0.05:
    print('Linear approximation is more accurate')

#Number 2
import scipy.integrate as spint

#Define the equations for the models
def compSim(y,t0,R1,R2,a11,a12,a21,a22):
    N1=y[0]
    N2=y[1]
    
    dN1dt=R1*(1-N1*a11-N2*a12)*N1
    dN2dt=R2*(1-N2*a22-N1*a21)*N2
    
    return [dN1dt,dN2dt]

#Model 1 - a12>a11 AND a21>a22 (neither criteria satisfied)
times=range(0,10)
y0=[1,5]
params=(0.8,0.8,0.5,2,2,0.5)
sim=spint.odeint(func=compSim,y0=y0,t=times,args=params)
simDF=pd.DataFrame({"t":times,"species1":sim[:,0],"species2":sim[:,1]})
ggplot(simDF,aes(x="t",y="species1"))+geom_line()+geom_line(simDF,aes(x="t",y="species2"),color='red')+theme_classic()


#Model 2 a12>a11 (first criterion not satisfied)
times=range(0,10)
y0=[1,5]
params=(0.8,0.8,0.5,2,0.5,2)
sim=spint.odeint(func=compSim,y0=y0,t=times,args=params)
simDF=pd.DataFrame({"t":times,"species1":sim[:,0],"species2":sim[:,1]})
ggplot(simDF,aes(x="t",y="species1"))+geom_line()+geom_line(simDF,aes(x="t",y="species2"),color='red')+theme_classic()


#Model 3 a21>a22 (second criterion not satusfied)
times=range(0,10)
y0=[1,5]
params=(0.8,0.8,2,0.5,2,0.5)
sim=spint.odeint(func=compSim,y0=y0,t=times,args=params)
simDF=pd.DataFrame({"t":times,"species1":sim[:,0],"species2":sim[:,1]})
ggplot(simDF,aes(x="t",y="species1"))+geom_line()+geom_line(simDF,aes(x="t",y="species2"),color='red')+theme_classic()


#Model 4 a12<a11 AND a21<a22 (both criteria satisfied)
times=range(0,10)
y0=[1,5]
params=(0.8,0.8,2,0.5,0.5,2)
sim=spint.odeint(func=compSim,y0=y0,t=times,args=params)
simDF=pd.DataFrame({"t":times,"species1":sim[:,0],"species2":sim[:,1]})
ggplot(simDF,aes(x="t",y="species1"))+geom_line()+geom_line(simDF,aes(x="t",y="species2"),color='red')+theme_classic()

