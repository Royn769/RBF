from typing import Any
import numpy as np
from RBF_supervised import RBF
from scipy.io import loadmat
from sklearn.metrics import r2_score
import time
from math import sqrt

start=time.time()

#%%
# normalize function
class MapMinMaxApplier(object):
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept
    def __call__(self, x):
        return x * self.slope + self.intercept
    def reverse(self, y):
        return (y-self.intercept) / self.slope
 
def mapminmax(x, ymin=-1, ymax=+1):
    x = np.asanyarray(x)
    if len(x[0])==1:
        xmax=x.max(axis=0)
        xmin=x.min(axis=0)
    else:
        xmax=x.max(axis=-1)
        xmin=x.min(axis=-1)
    slope = ((ymax-ymin) / (xmax - xmin))[:,np.newaxis]
    intercept = (-xmin*(ymax-ymin)/(xmax-xmin))[:,np.newaxis] + ymin
    ps = MapMinMaxApplier(slope, intercept)
    return ps(x), ps
#%%
# load data
data=loadmat('./matlab.mat')
T=np.transpose(np.array(data['T']))
T1=np.transpose(np.array(data['T1']))
U=np.transpose(np.array(data['U']))
U1=np.transpose(np.array(data['U1']))
#%%
# nromalize data
train_t,inputt=mapminmax(T1,0,1)
test_t=inputt(T)
RR=np.empty([15,1])
RRMSE=np.empty([15,1])
train_u,outputu=mapminmax(U1,0,1)
test_U=U
rbf=RBF(0.01,1000,10)
rbf.train(np.transpose(train_t),np.transpose(train_u))
pre_u=rbf.predict(np.transpose(test_t))
pre_U=outputu.reverse(np.transpose(pre_u))

for i in range(U.shape[0]):
    RR2=r2_score(test_U[i],pre_U[i])
    RR[i]=RR2
    RR2MSE=sqrt(np.sum((test_U[i] - pre_U[i])**2)/len(test_U[i]))/sqrt(np.sum((test_U[i]-np.mean(test_U[i]))**2)/(len(test_U[i])-1))
    RRMSE[i]=RR2MSE
print(RR)
# print(RRMSE)

end=time.time()
print('Running time: %s Seconds'%(end-start))