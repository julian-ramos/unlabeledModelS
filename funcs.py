'''
Created by Julian Ramos
CMU - 4/3/14
Model selection using unlabeled data
'''
from sklearn.datasets import  load_boston, load_iris, load_diabetes, load_digits, load_linnerud
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model,svm,tree
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score


def clfsEval(trainData,testData,valData,trainLabels,testLabels,valLabels):
    '''
    This function takes the data sets and build several classifiers
    then they are all evaluated
    '''
    clf={'classifier':[],'pred_test':[],'f1_score_test':[],'pred_val':[],'f1_score_val':[]}
    clfs=[GaussianNB(),linear_model.LogisticRegression(),svm.SVC(),tree.DecisionTreeClassifier(),SGDClassifier()]
    
    for i in range(len(clfs)):
        clf['classifier'].append(clfs[i])
        tempClf=clf['classifier'][i]
        tempClf.fit(trainData, trainLabels)
        clf['pred_test'].append(tempClf.predict(testData))
        clf['f1_score_test'].append(f1_score(clf['pred_test'][i],testLabels))
        clf['pred_val'].append(tempClf.predict(valData))
        clf['f1_score_val'].append(f1_score(clf['pred_val'][i],valLabels))
    
    #Here work on getting the agreementa rates for all of the classifiers
    
    return clf
        
        
    