# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 15:37:06 2017

@author: Administrator
"""

import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt

dt=0.0001
Vin=1


r1=0.100
g1=1/r1
c1=1e-6
vd=0.0001
gd=1e-12*np.exp(vd/0.0259)/0.0259
dgd=1e-12*np.exp(vd/0.0259)/0.0259/0.0259
                
X=np.zeros([3,1])
E=np.array([[g1,-g1,1],[-g1,g1+gd,0],[1,0,0]])
f=np.zeros([3,1])

nn=101
xtotal=np.zeros([3,nn])

for i in range(nn):
    f[-1]=+0.01*i    
    x=np.dot(np.linalg.inv(E),f)

    
    for k in range(30):
        F=np.dot(E,x)-f
    
        P=np.zeros([3,3])
        P[1,1]=dgd
     
        J=np.diag(np.dot(P,x)[:,0])+E
        y=np.dot(np.linalg.inv(J),F)
    
        x=x-y
                              
        err=np.linalg.norm(x)
        
        xtotal[:,i]=x.ravel()
         
        
        if(err<1e-4):
            print(k,err,x[0],x[1],x[2])
            break
        
        vd=x[1]
        gd=1e-12*np.exp(vd/0.0259)/0.0259
        dgd=1e-12*np.exp(vd/0.0259)/0.0259/0.0259
        E=np.array([[g1,-g1,1],[-g1,g1+gd,0],[1,0,0]])

plt.figure(1)
t = np.arange(nn)
plt.plot(t, xtotal[1-1][:], marker='', label='i1', markevery=2)
plt.plot(t, xtotal[2-1][:], marker='', label='i1', markevery=2)
plt.figure(2)
plt.plot(xtotal[1-1][:], xtotal[2-1][:], marker='', label='i1', markevery=2)
plt.figure(3)
plt.plot(xtotal[1-1][:], -xtotal[3-1][:], marker='', label='i1', markevery=2)


#
#
#nn=101
#Xtotal=np.zeros([3,nn])
#
#for i in range(nn):
#  
#    
#    Xtotal[:,i]=Xn.ravel()
#    Xn = np.dot(M1,X) + np.dot(M2,f)
#    X=Xn
#    f[-1]=1
#
#t = np.arange(nn)
#
#plt.figure(1)
#plt.plot(t, Xtotal[1-1][:], marker='', label='i1', markevery=2)
#plt.plot(t, Xtotal[2-1][:], marker='', label='i1', markevery=2)
##plt.plot(t, Xtotal[3-1][:], marker='', label='i1', markevery=2)
#plt.figure(2)
#plt.plot(t, -Xtotal[3-1][:], marker='', label='i1', markevery=2)
 
#
#
#nn=501
#Utotal=np.zeros([5+5+2,nn])
#for i in range(nn):
#    
#    Is5=0
#    Istep=0.00001
#    r1=1000
#    r3=3000*(1 + 0*(Utotal[5+3-1,i-1]**2 +1.5*Utotal[5+3-1,i-1]**1))
#    r4=4000
#    G2=0.0001
#    
#    Ki=np.array([r1,1,r3,r4,1])
#    Ki=np.diag(Ki)
#    Kv=np.array([-1,-G2,-1,-1,0])
#    Kv=np.diag(Kv)
#    Kv[1][2]=Kv[1][1]
#    Kv[1][1]=0
#    
#    s=np.zeros([5+5+2,1])
#    s[-1]=Is5 + i*Istep
#    
#    T=np.zeros([2+5+5,5+5+2])
#    T[  0   :   A.shape[0]          ,   0       :  A.shape[1]        ] = A
#    T[  2   :   2+5                 ,   5       :  5+5               ] = I
#    T[  2   :   2+AT.shape[0]       ,   5+5     :  5+5+AT.shape[1]   ] = -AT
#    T[  2+5 :   2+5+Ki.shape[1]     ,   0       :  0+Ki.shape[1]     ] = Ki
#    T[  2+5 :   2+5+Kv.shape[1]     ,   5       :  5+Kv.shape[1]     ] = Kv
#    #print(np.linalg.det(T)) 
#    U=np.dot(np.linalg.inv(T),s)
#    Utotal[:,i]=U.ravel()
#         
#x = np.arange(nn)
#plt.plot(x, Utotal[1-1][:], marker='', label='i1', markevery=2)
#plt.plot(x, Utotal[2-1][:],  marker='', label='i2' , markevery=2)
#plt.plot(x, Utotal[3-1][:],  marker='', label='i3' , markevery=2)
#plt.plot(x, Utotal[4-1][:],  marker='', label='i4' , markevery=2)
#plt.plot(x, Utotal[5-1][:],  marker='', label='i5' , markevery=2)
#plt.xlabel("nn")
#plt.ylabel("Current (A)")
##plt.ylim(0, 1.0)
#plt.legend(loc='lower right')
#plt.show()
# 
##T=np.zeros([5+5+2,5+5+2])
##T[  0   :   A.shape[0]     ,   0       :  A.shape[1]        ] = A
##T[  5   :   5+5            ,   5       :  5+5               ] = I
##T[  5   :   5+AT.shape[0]  ,   5+5     :  5+5+AT.shape[1]   ] = -AT
##T[  5+5 :   5+5+1          ,   0       :  0+Ki.shape[1]     ] = Ki
##T[  5+5 :   5+5+1          ,   5       :  5+Kv.shape[1]     ] = Kv
##T[  5+5+1 :   5+5+1+1      ,   5       :  5+Kv.shape[1]     ] = Kv
###  so far T is a singular matrix
##print(np.linalg.det(T)) 
##U=np.dot(np.linalg.inv(T),s)
