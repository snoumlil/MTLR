# import
from __future__ import division
from scipy import *
import numpy as np 
from scipy.optimize import *
from cvxopt import *
from pylab import *
import matplotlib.pyplot as plt
import csv
import argparse
import bigfloat as bf
import timeit
from scipy import stats


#Write the weights: np.savetxt("lol.csv",a,delimiter=" ")
# read the weights : np.loadtxt("lol.csv")

# valide
# Input  filename : the name of the file
# Output  nd array of covariates
def covariates(fileName):
	liste=[]
	fichier=open(fileName, 'rb')
	reader = csv.reader(fichier, delimiter=',')
	for row in reader:
		liste.append(row)
	for i in xrange(0,shape(liste)[0]):
		liste[i].pop(0)
	liste.pop(0)
	for i in xrange(0,shape(liste)[0]):
		for j in xrange(0,shape(liste[0])[0]):
			liste[i][j]=float(liste[i][j])
	fichier.close()
	return np.array(liste)


#Valide
# Compute the zcore of each column
def standardize(covariates):
	a=covariates.transpose()
	for i in xrange(0,shape(a)[0]):
		a[i]=stats.zscore(a[i])		
	return a.transpose()



#Valide
# Input : filename : the name of the csv file 
# Output : An array of survival Times	
def readSurv(fileName):
	return np.genfromtxt(fileName,usecols=(0),skip_header=1,delimiter=',',dtype=None)


# Valide
# Inputs: -survivalTimes : Array of survival times 
#         - nbPoints : number of points we want 
# Output: a nbPoints lentght array from the 1st to the 100 th percentile of survival times.
#     timepoints=[t1,t2,.....tnbPoints]
def timePoints(survivalTimes,nbPoints):
	time=[]
	for i in arange(0,100,100/nbPoints):
		time.append((np.percentile(survivalTimes,i)))
	return array(time)

#Valide
# Inputs :  - survTime : survival Time 
#           - time : array of timePoints
# Outputs:  a binary sequence of survival time
def encode(survTime,timePoints):
	return (timePoints>=survTime).astype(int)
	
#valide
# Compute the binary sequence for all survival times
def computeY(survivalTimes,timePoints):
	nbTime=shape(timePoints)[0]
	nPatient=shape(survivalTimes)[0]
	Y=np.zeros((nPatient,nbTime))
	for i in xrange(0,nPatient):
		Y[i]=encode(survivalTimes[i],timePoints)
	return Y

# Inputs: -Theta: ndArray (initial weights)
#         - xi array of covariates for patient i
# 		  -  k integer
# Outputs: - The score of the sequence with the event occuring in the interval [tk,tk+1)
def fscore(Theta,b,xi,k):
	maxMonth=shape(Theta)[0]
	return (Theta[k+1:maxMonth]*xi).sum()+b[k+1:maxMonth].sum()

# valide
def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"

#Valide
# Inputs weigths=[Theta b]
#        X : array of covariates
#        nbTime,nVar,nPatient
#        c1 and c2 parameters
# Outputs cost	
def cost(Weights,X,Y,c1,c2,nbTime,nVar,nPatient):
	Weights=Weights.reshape(nbTime,nVar+1)
	X=X.reshape(nPatient,nVar)
	b=Weights[:,nVar]
	theta=Weights[:,0:nVar]
	#Expression 1	
	s1=0
	for j in xrange(0,nbTime):
		s1=s1+norm(theta[j],2)**2
	#Expression2
	s2=0
	for j in xrange(0,nbTime-1):
		s2=s2+(c2/2)*norm(theta[j+1]-theta[j],2)**2
	#s2=norm(t[1:]-t[:-1])**2
	#Expression3
	s3=0
	for i in xrange(0,nPatient):
		s31=0
		s32=0
		for j in xrange(0,nbTime):
			s31=s31+Y[i][j]*(np.dot(theta[j],X[i].T)+b[j])
		for k in xrange(-1,nbTime):
			s32=s32+(exp(fscore(theta,b,X[i],k)))
		s3=s3+s31-log(s32)	
	return (s1+s2-(c1/nPatient)*s3)



# Valide
# Compute the likelihood of a patient
def likelihood(theta,b,x,ti,time):
	y=encode(ti,time)
	nbTime=shape(theta)[0]	
	num=0
	den=0
	for i in xrange(0,nbTime):
		num=num+y[i]*(np.dot(theta[i],x.T)+b[i])
	for k in xrange(-1,nbTime):
		den=den+exp(fscore(theta,b,x,k))	
        return (exp(num)/den)

#Valide
# Compute the likelihood for a censored patient
def likelihoodCensored(topt,b,x,ti,timePoints):
        [nbTime,nVar]=shape(topt)
        num=0
        den=0
        for k in xrange(findNearest(timePoints,ti),nbTime):
                num=num+exp(fscore(topt,b,x,k))
        for s in xrange(0,nbTime):
                den=den+exp(fscore(topt,b,x,s))
        return (num/den)
       
#Valide
# Input array : An array of time points ; value : a value
# Output  the index of closest value which is in  the array          	
def findNearest(array,value):
    idx=(np.abs(array-value)).argmin()
    return idx 


# cost 2
def cost2(Weights,X,surv,time,nbTime,nVar,nPatient):
    X=X.reshape(nPatient,nVar)
    Weights=Weights.reshape(nbTime,nVar+1)
    b=Weights[:,nVar]
    t=Weights[:,0:nVar]
    s=0
    for i in xrange(0,nPatient):
                s=s+log(likelihoodCensored(t,b,X[i],surv[i]))
    return -s




# Valide
def absErrorAE(predictedTime,survivalTime):
	return abs(bf.log(predictedTime)-bf.log(survivalTime))

def l(p,t):
	return min(abs((p-t)/p),1)


def l2(p,t):
	return abs(p-t)

#Valide
# predTime
def predTime(topt,b,x,timePoints):
	nbTime=shape(timePoints)[0]
	res=np.zeros(nbTime)
	for j in xrange(0,nbTime):
		s=0
		for i in xrange(0,nbTime):
			s=s+absErrorAE(timePoints[j],timePoints[i])*likelihood(topt,b,x,timePoints[i],timePoints)
		res[j]=s
	return timePoints[res.argmin()]


#Valide
def deathTimes(topt,b,X,timePoints):
	l=[]	
	for i in xrange(0,shape(X)[0]):
		l.append(predTime(topt,b,X[i],timePoints))
	return array(l)


     

#Valide
# error
def rateError(surv,death,tolerence):
	num=0
	den=shape(death)[0]
	for i in arange(0,shape(death)[0]):
		if(abs(surv[i]-death[i])>=tolerence):
			num=num+1
	return (num/den)*100



# Approximate derivative of the cost
def costDerW(W,X,Y,c1,c2,nbTime,nVar,nPatient):
	W=W.reshape(nbTime,nVar+1)
	X=X.reshape(nPatient,nVar)
	dW=np.ones((nbTime,nVar+1))/10000000000
	return ((cost(W+dW,X,Y,c1,c2,nbTime,nVar,nPatient)-cost(W,X,Y,c1,c2,nbTime,nVar,nPatient))/dW).flatten()


# train 
def  trainMtlr(X,surv,nbPatient,nbTimePoints,c1,c2):
	X=X[0:nbPatient]
	surv=surv[0:nbPatient]
	nVar=shape(X)[1]
	time=timePoints(surv,nbTimePoints)
	W=np.random.random_sample((nbTimePoints,nVar+1))
	Y=computeY(surv,time)
	# Avec precalcul
	opt0 = minimize(cost,W,args=(X,Y,c1,c2,nbTimePoints,nVar,nbPatient),jac=costDerW,method="Newton-CG",options={'maxiter': 10000000})
	tic()
	opt = minimize(cost,opt0.x,args=(X,Y,c1,c2,nbTimePoints,nVar,nbPatient),method="SLSQP",options={'maxiter': 400000})
	toc()
	T=opt.x
	T=T.reshape(shape(W))
	return T	




# Compute the error
def computeError(X,surv,opt,nbPatient,nbTimePoints,tolerence):	
	X=X[0:nbPatient]	
	nVar=shape(X)[1]
	surv=surv[0:nbPatient]
	time=timePoints(surv,nbTimePoints)
	b=opt[:,nVar]
	topt=opt[:,0:nVar]
	death=deathTimes(topt,b,X,time)
	return rateError(surv,death,tolerence)


# predict survival time
def predict(X,surv,opt,nbPatient,nbTimePoints):
	nVar=shape(X)[1]
	X=X[0:nbPatient]
	surv=surv[0:nbPatient]
	b=opt[:,nVar]
	topt=opt[:,0:nVar]
	time=timePoints(surv,nbTimePoints)
	death=deathTimes(topt,b,X,time)
	print "True survival"+"      "+"Predicted "+"\n"
	for i in xrange(0,shape(surv)[0]):
		print str(surv[i]) +"   "+ str(death[i])
	return death

#Valide
# Plot the survival
def plotSurvival(topt,b,X,Y,NbPatient,maxMonth,time):
	nbTime=shape(topt)[0]
	nbPatient=shape(X)[0]
	Time=[]
	for i in xrange(0,maxMonth):
		Time.append(i)
	AllPatient=[]
	temp=[]
	for i in xrange(0,NbPatient):
		for j in xrange(0,maxMonth):
			a=likelihoodCensored(topt,b,X[i],j,time)
			temp.append(a)
		AllPatient.append(temp)
		temp=[]
	for i in xrange(0,NbPatient):
		plt.plot(Time,AllPatient[i],label=str(i))
		plt.legend(loc='uper left')
	show()
	return 1


