from module import *

#Parameters
nbPatient=800
nbTimePoints=50
c1=1000
c2=1
filename="class.csv"
X=standardize(covariates(filename))
surv=readSurv(filename)



#Classification
Xclass=X[0:2000]
survClass=surv[0:2000]


opt=trainMtlr(X,surv,nbPatient,nbTimePoints,c1,c2)
write(opt,"topt.csv")# 800 par 40

death=predict(X,surv,opt,nbPatient,nbTimePoints,1)

tolerence=3 # difference tolerated between the true and real time
computeError(X,surv,opt,nbPatient,nbTimePoints,tolerence)

write(opt,"topt.csv")# 500 par 30


# Test 
#2050
Xtest=X[2000:2120]
survTest=surv[2000:2120]

nbPatientTest=150
Predicted=predict(Xtest,survTest,opt,nbPatientTest,nbTimePoints,1)

errorTest=computeError(Xtest,survTest,opt,nbPatientTest,nbTimePoints,tolerence)
errorTest
