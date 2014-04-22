'''
Created by Julian Ramos
CMU - 4/3/14
Model selection using unlabeled data
'''
import numpy as np
import dataStats as dS
from sklearn.datasets import  load_boston, load_iris, load_diabetes, load_digits, load_linnerud
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model,svm,tree
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt
from sklearn.metrics import classification_report



def clfsEval(trainData,testData,valData,trainLabels,testLabels,valLabels,uLabels):
    '''
    This function takes the data sets and builds several classifiers
    then they are all evaluated
    '''
#     uLabels=np.hstack((np.unique(trainLabels),np.unique(testLabels),np.unique(valLabels)))
#     uLabels=np.unique(uLabels)
    
    clf={'classifier':[],'pred_test':[],'f1_score_test':[],'pred_val':[],\
         'f1_score_val':[],'confusion_matrix_test':[],'confusion_matrix_val':[],\
         'report_test':[],'report_val':[]}
    clfs=[GaussianNB(),
          linear_model.LogisticRegression(),
          svm.SVC(),
          tree.DecisionTreeClassifier(),
          SGDClassifier()]
    
    for i in range(len(clfs)):
        clf['classifier'].append(clfs[i])
        tempClf=clf['classifier'][i]
        tempClf.fit(trainData, trainLabels)
        clf['pred_test'].append(tempClf.predict(testData))
        clf['f1_score_test'].append(f1_score(testLabels,clf['pred_test'][i],labels=uLabels))
        clf['confusion_matrix_test'].append(confusion_matrix(testLabels,clf['pred_test'][i]))
        clf['report_test'].append(classification_report(testLabels,clf['pred_test'][i]))
        clf['pred_val'].append(tempClf.predict(valData))
        clf['f1_score_val'].append(f1_score(valLabels,clf['pred_val'][i],labels=uLabels))
        clf['confusion_matrix_val'].append(confusion_matrix(valLabels,clf['pred_val'][i]))
        clf['report_val'].append(classification_report(valLabels,clf['pred_val'][i]))

    
    return clf

def agreementRates(clf,valLabels,uLabels,plot=False):
    '''
    This function gets a classifiers dictionary generated from clfsEval
    here we are going to calculate the agreement of the classifiers and
    generate a list of accuracy, f1_scores and other performance metrics
    so that we can figure out the thold that can maximize performance
    '''
    
    clfsLabels=[i for i in clf['pred_val']]
    clfsLabels=np.array(clfsLabels)
    
    agmnt=[]
    agmntVal=[]
    for i in range(len(clfsLabels[1])):
        temp=clfsLabels[:,i]
        mx=len(temp)
        cts=dS.counts(temp)
        oCts=sorted(cts['counts'])
        temp=dS.expSmooth(oCts,0.95)/mx
        agmnt.append(temp)
        
        
        tmax=max(cts['counts'])
        tind=np.argwhere(np.array(cts['counts'])==tmax)
        
        #Check whether there is a tie in the number of counts
        #if there is one simply throw the dies
        if len(tind)==1:
            tind=np.argmax(cts['counts'])
            agmntVal.append(cts['vals'][tind])
        else:
            tind=np.random.permutation(tind)[0]
            agmntVal.append(cts['vals'][tind])
            
        
            
    
    sAgmnt=np.sort(np.unique(agmnt))
    f1_scores=[]
    for i in sAgmnt:
        inds2=np.argwhere(agmnt>=i)
        print(len(inds2),' size data ')
        temp=[]
        #Now that I have this I can generate the label
        #for each data point based on the highest label
        #agreed on. Once I obtain these labels I can rank 
        #the algorithms and then I can compare this ranking
        #to the one produced by the real labels
        
        for cnum in range(len(clf['pred_val'])):
            try:
                temp.append(f1_score(valLabels[inds2],clf['pred_val'][cnum][inds2],labels=uLabels))
            except:
                print('here')
            print(f1_score(valLabels[inds2],clf['pred_val'][cnum][inds2],labels=uLabels))
        f1_scores.append(temp)
    if plot==True:
        plt.plot(sAgmnt,f1_scores)
        plt.show()
    
#     for i in range(len(clf['pred_val'])):
#         print(f1_score(valLabels,clf['pred_val'][i]))
#         print(clf['f1_score_val'][i])
#         print(f1_score(valLabels,clf['pred_val'][i])==clf['f1_score_val'][i])
#     print('here')
    return agmntVal,agmnt
    
        
def clfsVal(clfs,agmntLabels,agmntLevels,uLabels,agmntLvl=1):
    '''
    Take the classifiers and the already calculated outputs and measure their f1_score
    for the labels generated from the agreement step
    '''
    clfs['f1_score_agmnt']=[]
    for i in range(len(clfs['pred_val'])):
        inds=np.argwhere(np.array(agmntLevels)>=agmntLvl)
        agmntF1=f1_score(np.array(agmntLabels)[inds],np.array(clfs['pred_val'][i])[inds],labels=uLabels)
        clfs['f1_score_agmnt'].append(agmntF1)
        
    
    
    

        

    
    
    
    
    