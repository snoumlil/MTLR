from mtlr import *

"""loading the data"""
"""The first column should refer to survival times, the others to the covariates"""
filename="data.csv"

X=covariates(filename)
X=standardize(X)

survivalTimes=readSurv(filename)

""" We choose only 10 patient  to make the computing quick for this exemple"""
X=X[0:10]
survivalTimes=survivalTimes[0:10]



"""Getting the parameters"""
nPatient=shape(X)[0]
nVar=shape(X)[1]

"""Creating the timePoints vector""" 
nbTime=5
timePoints=timePoints(survivalTimes,nbTime)

"""Computing the Binary sequences for all survival times"""
Y=computeY(survivalTimes,timePoints)


"""Computing the cost""" 
initialWeights=np.random.random_sample((nbTime,nVar+1)) # initialWeights=[theta b]
c1=1000
c2=1

cost(initialWeights,X,Y,c1,c2,nbTime,nVar,nPatient)


"""Minimizing the cost""" 
opt = minimize(cost,initialWeights,args=(X,Y,c1,c2,nbTime,nVar,nPatient),method="SLSQP",options={'maxiter': 400000})

optimalWeights=opt.x
optimalWeights=optimalWeights.reshape(nbTime,nVar+1)

thetaOpt=optimalWeights[:,0:nVar]
bOpt=optimalWeights[:,nVar]


"""Predicting the survival time of a patient i with covariates X[i]"""
i=1
predicted=predTime(thetaOpt,bOpt,X[i],timePoints)
true=survivalTimes[i]



"""Predicting the survival times for all patients"""
PredictedSurvivalTimes=deathTimes(thetaOpt,bOpt,X,timePoints)

"""computing rate error between the predicted times and the true ones (%)"""
tolerence=3
rateError(survivalTimes,PredictedSurvivalTimes,tolerence)

"""Computing the likelihood of the patient i with covariate X[i] at time t"""
t=3
likelihood(thetaOpt,bOpt,X[1],t,timePoints)
likelihoodCensored(thetaOpt,bOpt,X[1],t,timePoints)

"""Plotting  the Survival Probabilty of N patient"""
N=10 # number of patient to plot
maxMonth=30
plotSurvival(thetaOpt,bOpt,X,Y,N,maxMonth,timePoints)


#######################################################################################
# to use 

"""train mtlr"""
optimalWeights=trainMtlr(X,survivalTimes,nPatient,nbTime,c1,c2)

"""predict survival times""" 
predictedTimes=predict(X,survivalTimes,optimalWeights,nPatient,nbTime)

"""Compute error"""
tolerence=3
computeError(X,survivalTimes,optimalWeights,nPatient,nbTime,tolerence)








