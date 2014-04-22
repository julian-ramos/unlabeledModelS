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


while cont<expe['numTests']:
    
    try:
        #Divide the data in training, testing and validation

        splitData=dmlPre.dataSplitBalancedClass(data, labels)
        trainData=splitData['trainData']
        trainLabels=splitData['trainLabels']
        testData=splitData['testData']
        testLabels=splitData['testLabels']
        valData=splitData['valData']
        valLabels=splitData['valLabels']
        uLabels=np.unique(np.hstack((np.unique(trainLabels),np.unique(testLabels),np.unique(valLabels))))
        
        clfs=fun.clfsEval(trainData,testData,valData,trainLabels,testLabels,valLabels,uLabels)
        agmntLabels,agmntLevels=fun.agreementRates(clfs,valLabels,uLabels,plot=True)
        fun.clfsVal(clfs,agmntLabels,agmntLevels,uLabels,agmntLvl=expe['agmntlvl'])
        f1_score_agmnt=f1_score(valLabels,agmntLabels,labels=uLabels)
        confusion_agmnt=confusion_matrix(valLabels,agmntLabels)
        
        if expe['verbose']==1:
            print('Agreement level used for filtering = %d'%(expe['agmntlvl']))
            print('summary of results validation data')
            print(clfs['report_val'][0])
            
            print('summary of results agreement data')
            print(classification_report(valLabels,agmntLabels))
            
            print('f1 score test data',clfs['f1_score_test'])
            print('f1 score validation data',clfs['f1_score_val'])
            print('f1 score agmnt data',clfs['f1_score_agmnt'])
            # print(clfs['confusion_matrix_val'],'confusion matrix val')
            # print(confusion_agmnt,'confusion matrix agmnt')
            print('f1_score validation and agmntLabels',f1_score_agmnt)
        
        
        #In this section I have to consider when there are 
        #ties between multiple classifiers
        rVal=np.argsort(clfs['f1_score_val'])[::-1]
        rAgmnt=np.argsort(clfs['f1_score_agmnt'])[::-1]
        
        rankVal=range(len(rVal))
        rankAgmnt=[int(np.argwhere(rVal==i)) for i in rAgmnt]

        rankVal=np.array(rankVal)+1
        rankAgmnt=np.array(rankAgmnt)+1
        spearmans=spearmanr(rankVal,rankAgmnt)
        
        if expe['verbose']==1:
            print('models ordered by f1_score')
            print(rVal)
            print(rAgmnt)
            print('models ranking')
            print(rankVal)
            print(rankAgmnt)
            print(spearmans,'spearman rank correlation coefficient' )
        
        
        spear.append(spearmans)
        cont+=1
    except:
        print('Failed trying again')
print spear


# It seems like everything is working fine, what I need to do next is:
# Calculate for everyrun the spearmans for different levels of agmnt
# acumulate those results and then plot them with error whiskers, create a function to 
# create that graph