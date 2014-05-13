'''
Created by Julian Ramos
CMU - 4/3/14
Model selection using unlabeled data
'''
from scipy.stats import spearmanr
import numpy as np
import dataML_Prepro as dmlPre
from sklearn.datasets import  load_boston, load_iris, load_diabetes, load_digits, load_linnerud
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.metrics import f1_score
import funcs as fun
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import General as gF


#Datasets

expe=gF.experimentVariables('unlabeledModelS')
exec(expe['variables'])

data=expe['data']
labels=expe['labels']
cont=0
ranks=[]
f1_score_mv_predval_agmnt=[]
oldAgmntLevels=[]
bestF1s=[]

while cont<expe['numTests']:
    
    #Divide the data in training, testing and validation
    splitData=dmlPre.dataSplitBalancedClass(data, labels)
    trainData=splitData['trainData']
    trainLabels=splitData['trainLabels']
    testData=splitData['testData']
    testLabels=splitData['testLabels']
    valData=splitData['valData']
    valLabels=splitData['valLabels']
    uLabels=np.unique(np.hstack((np.unique(trainLabels),np.unique(testLabels),np.unique(valLabels))))
    
    #Train the different classifiers and predict for the testing and validation data sets
    clfs=fun.clfsEval(trainData,testData,valData,trainLabels,testLabels,valLabels,uLabels)
    
    #Calculates the majority vote and the agreement rate for all of the classifiers
    mvLabels,agmntLevels=fun.agreementRates(clfs,valLabels,uLabels,plot=expe['plots'])
    if oldAgmntLevels==[]:
        oldAgmntLevels=sorted(np.unique(agmntLevels))
    
    if oldAgmntLevels!=[]:
        if sorted(np.unique(agmntLevels))==oldAgmntLevels:
            
            #Calculates the f1_scores for the majority vote labels and the predicted outputs
            #from the different classifiers
            fun.clfsVal(clfs,mvLabels,agmntLevels,uLabels)
            f1_score_agmnt=f1_score(valLabels,mvLabels,labels=uLabels)            
            
            if expe['verbose']==1:
                confusion_agmnt=confusion_matrix(valLabels,mvLabels)
                print('Agreement level used for filtering = %d'%(expe['agmntlvl']))
                print('summary of results validation data')
                print(clfs['report_val'][0])
                
                print('summary of results agreement data')
                print(classification_report(valLabels,mvLabels))
                
                print('f1 score test data',clfs['f1_score_test'])
                print('f1 score validation data',clfs['f1_score_val'])
                print('f1 score agmnt data',clfs['f1_score_agmnt'])
                # print(clfs['confusion_matrix_val'],'confusion matrix val')
                # print(confusion_agmnt,'confusion matrix agmnt')
                print('f1_score validation and mvLabels',f1_score_agmnt)
                print('models ordered by f1_score')
                print(rVal)
                print(rAgmnt)
                print('models ranking')
                print(rankVal)
                print(rankAgmnt)
            
            fun.spearmansCalc(clfs)    
                
            spear.append(clfs['spearmans'])
            ranks.append(clfs['rank_mv_predval_agmnt'])
            tempBestMods=[ i[0] for i in clfs['rank_mv_predval_agmnt']]
            tempBestF1s=[clfs['f1_score_val_predval'][i] for i in tempBestMods]
            bestF1s.append(tempBestF1s)
            
            cont+=1
            print('iteration',cont)
            
            signal2plot=expe['signal2plot']
            if f1_score_mv_predval_agmnt==[]:
                f1_score_mv_predval_agmnt=[[] for i in range(len(clfs[signal2plot]))]
            for i in range(len(clfs[signal2plot])):
                f1_score_mv_predval_agmnt[i]=f1_score_mv_predval_agmnt[i]+[clfs[signal2plot][i]]
        else:
            print('The experiment didnt produce similar agreement levels')
            print(oldAgmntLevels)
            print(agmntLevels)
    
# valRank=clfs['']
fun.summaryGraphs(f1_score_mv_predval_agmnt,len(clfs['classifier']),spear,ranks,bestF1s,sorted(np.unique(agmntLevels)))



# It seems like everything is working fine, what I need to do next is:
# Calculate for everyrun the spearmans for different levels of agmnt
# acumulate those results and then plot them with error whiskers, create a function to 
# create that graph