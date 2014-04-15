'''
Created by Julian Ramos
CMU - 4/3/14
Model selection using unlabeled data
'''
from scipy.stats import spearmanr
import numpy as np
from sklearn.datasets import  load_boston, load_iris, load_diabetes, load_digits, load_linnerud
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.metrics import f1_score
import funcs as fun

datasets={'boston':load_boston(),'iris':load_iris(),'diabetes':load_diabetes(),'digits':load_digits(),'linnerud':load_linnerud()}

data=datasets['digits']['data']#[0:135,:]
labels=datasets['digits']['target']#[0:135]

#Divide the data in training, testing and validation

allInds=np.arange(len(data))
np.random.shuffle(allInds)

chunkS=len(data)/3
trainInds=allInds[0:chunkS]
testInds=allInds[chunkS:chunkS*2]
valInds=allInds[chunkS*2:]

trainData=data[trainInds]
trainTarget=labels[trainInds]
testData=data[testInds]
testTarget=labels[testInds]
valData=data[valInds]
valTarget=labels[valInds]

clfs=fun.clfsEval(trainData,testData,valData,trainTarget,testTarget,valTarget)
agmntLabels,agmntLevels=fun.agreementRates(clfs,valTarget)
fun.clfsVal(clfs,agmntLabels,agmntLevels,agmntLvl=0.0)
f1_score_agmnt=f1_score(valTarget,agmntLabels)
print(clfs['f1_score_test'],'f1 score test data')
print(clfs['f1_score_val'],'f1 score validation data')
print(clfs['f1_score_agmnt'],'f1 score agmnt data')
print(f1_score_agmnt,'f1_score validation and agmntLabels')

#IN this section I have to consider when there are 
#ties between multiple classifiers


rVal=np.argsort(clfs['f1_score_val'])[::-1]
rAgmnt=np.argsort(clfs['f1_score_agmnt'])[::-1]

rankVal=range(len(rVal))
rankAgmnt=[int(np.argwhere(rVal==i)) for i in rAgmnt]

print(rVal)
print(rAgmnt)

rankVal=np.array(rankVal)+1
rankAgmnt=np.array(rankAgmnt)+1

print(rankVal)
print(rankAgmnt)
print(spearmanr(rankVal,rankAgmnt))



# Now I can rank the classifiers using the validation data and validation results
# and compare that ranking to the one generated with the agmntLabels with those two
# ranks I can use the spearman's ranking to compute whether the two rankings are
# correlated or not I could also use the kappa's statistic to measure this



# print(clfs)
# #Naive Bayes model
# gnb=GaussianNB()
# y_gnb = gnb.fit(trainData, trainTarget).predict(testData)
# #Linear regression
# lgr = linear_model.LogisticRegression()
# y_lgr = lgr.fit(trainData,trainTarget).predict(testData)
#  
# acuGnb=(testTarget == y_gnb).sum()/float(len(testTarget))
# print("Accuracy Naive Bayes on test data : %f" % acuGnb)
# f1Gnb=f1_score(testTarget,y_gnb)
# print("f1_score Naive Bayes on test data : %f" % f1Gnb)
# 
# acuLgr=(testTarget == y_lgr).sum()/float(len(testTarget))
# f1Lgr=f1_score(testTarget,y_gnb)
# print("Accuracy Logistic regression on test data : %f" % acuLgr)
# print("f1_score Logistic regression on test data : %f" % f1Lgr)
# 
# 
# 
# #Here comes the evaluation of the models with unlabeled data
# val_gnb=gnb.predict(valData)
# val_lgr=lgr.predict(valData)
# 
# ag=val_gnb==val_lgr
# 
# 
# indsAg=[i for i in range(len(ag)) if ag[i]==True]
# val_ag=val_gnb[indsAg]
# 
# print('accuracy on the agreed data',sum(val_ag==valTarget[indsAg])/float(len(val_ag)))
# print('f1_score on the agreed data',f1_score(val_ag,valTarget[indsAg]))
# print('data points used ',len(val_ag),' out of ',len(ag))



