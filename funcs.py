'''
Created by Julian Ramos
CMU - 4/3/14
Model selection using unlabeled data
'''
import dataStats as dS
import dataSearchUtils as dSu
import dataReader as dR
from scipy.stats import spearmanr
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


def izbickiSternScore(majorityVote,modelsIndex,realRanks):
    '''
    Calculate here the score described in Izbicki and stern paper
    Learning with many experts: Model selection and sparsity
    '''
    score=[]
    ranks=[]
    spears=[]
    modelsIndex=np.array(modelsIndex)
    realRanks=np.array(realRanks)
    for i in range(len(majorityVote)):
        score.append([])
        for i2 in range(np.shape(majorityVote[i])[1]):
            temp=majorityVote[i][:,i2].tolist()
            temp2=[sum([1 for i4 in temp if i3!=i4]) for i3 in temp]
            score[-1].append(temp2)
        sums=np.sum(score[-1],0)
        ranksIndex=np.argsort(sums)
        tempRank=modelsIndex[i,ranksIndex]
        ranks.append(tempRank.tolist())
        
        
        rank1,rank2=monotonicRanker(realRanks[i,:],tempRank)
#         spears.append(spearmanr(realRanks[i,:],tempRank))
        spears.append(spearmanr(rank1,rank2))
        
        
        #Just need to finish a couple of things
        #Include the agreement rate as a way to filter out the data and check on the
        #effect it has on the IS score      
            
    
    return score,ranks,spears

def dataExtract(filename):
    '''
    Extracts the data for the activity recognition study
    '''
    
    tempData=dR.csvReader(filename)
    data=[]
    labels=[]
    
    features=tempData['header'][2][3:]
    
    for vals in tempData['data']:
        data.append([float(i) for i in vals[3:]])
        labels.append(int(vals[0]))
        
        
    #Excluding missing data and the missing features
    inds=dSu.listStrFind(features, 'm_')
    ind=min(inds)
    
    #Missing data extraction
    features=features[:ind]
    data=np.array(data)
    missingData=np.sum(data[:,ind:-1],1)
    rows=np.argwhere(missingData==8).ravel()
    data=data[rows,:ind]
    labels=np.array(labels)
    labels=labels[rows]
    return data,labels,features
    
        
    
    
    

def clfsEval(trainData,testData,valData,trainLabels,testLabels,valLabels,uLabels,classN=None):
    '''
    This function takes the data sets and builds several classifiers
    then they are all evaluated
    ClassN=1 means it will only run the specified classifier
    '''

    
    clf={'classifier':[],'pred_test':[],'f1_score_test_predtest':[],'pred_val':[],\
         'f1_score_val_predval':[],'confusion_matrix_test':[],'confusion_matrix_val':[],\
         'report_test':[],'report_val':[]}
    if classN==None:
        clfs=[GaussianNB(),
              linear_model.LogisticRegression(),
              svm.SVC(),
              tree.DecisionTreeClassifier(),
              SGDClassifier()]
    else:
        clfs=[linear_model.LogisticRegression()]
    
    for i in range(len(clfs)):
        clf['classifier'].append(clfs[i])
        tempClf=clf['classifier'][i]
        tempClf.fit(trainData, trainLabels)
        clf['pred_test'].append(tempClf.predict(testData))
        clf['f1_score_test_predtest'].append(f1_score(testLabels,clf['pred_test'][i],labels=uLabels))
        clf['confusion_matrix_test'].append(confusion_matrix(testLabels,clf['pred_test'][i]))
        clf['report_test'].append(classification_report(testLabels,clf['pred_test'][i]))
        clf['pred_val'].append(tempClf.predict(valData))
        clf['f1_score_val_predval'].append(f1_score(valLabels,clf['pred_val'][i],labels=uLabels))
        clf['confusion_matrix_val'].append(confusion_matrix(valLabels,clf['pred_val'][i]))
        clf['report_val'].append(classification_report(valLabels,clf['pred_val'][i]))

    
    return clf
def majorityVote(clfs,testingData,ignore=[]):
    '''
    This function takes a set of previously built classifiers and calculates the
    majority vote for a given testing data set as well some performance measures related with
    the majority vote itself
    ignore is a list of the indexes of the classifiers to be ignored
    '''
    votes=[]
    mV=[]
    agmnt=[]
    
    for i in range(len(clfs)):
        if i not in ignore:
            temp=clfs[i]['classifier'][0].predict(testingData)
            votes.append(temp)
    votes=np.array(votes)
    for i in range(np.shape(votes)[1]):
        temp=votes[:,i]
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
            mV.append(cts['vals'][tind])
        else:
            tind=np.random.permutation(tind)[0]
            mV.append(cts['vals'][tind])
            #Check the agreement and majority vote values
            #I should be done with this section if it is
            #working correctly
            
    return mV,agmnt,votes

def ranking(agmntRate,mVote,predlabels,labels,agmntLvl=0):
    '''
    gives back the rank of each classifier for an agreement level
    less or equal than agmntLvl
    '''
    f1_scores=[]
    #Stores the indices of the filtered out data by agmnt level
    indsAgmnt=[]
    #Indeces of the classifiers used in the experiment
    #This should be in the output of the function but it is
    #not
    inds=[]
    ranks=[]
    realRanks=[]
    realF1_scores=[]
    spears=[]
    uLabels=[np.unique(i).tolist() for i in labels]
    uLabels=np.unique([i2 for i in uLabels for i2 in i])
    #Go through each of the experiments
    for expId in range(len(predlabels)):
        #Go through each of the classifiers
        tempF1_score=[]
        tempF1_scoreReal=[]
        for clfId in range(np.shape(predlabels[expId])[0]):
            #Changing this condition changes the entire behavior of the filtering
            #by agreement level
            indsAgmnt=np.argwhere(agmntRate[expId]<=agmntLvl)
            try:
                tempF1_score.append(f1_score(np.array(mVote[expId])[indsAgmnt],predlabels[expId][clfId,indsAgmnt],labels=uLabels))
            except:
                print('here')
            tempF1_scoreReal.append(f1_score(labels[expId][indsAgmnt],predlabels[expId][clfId,indsAgmnt],labels=uLabels))
            # Not sure whether I should compare against the ranking generated by the full data set or
            # the filtered data set??? 
        temp=[]
        
        for i in range(np.shape(predlabels[expId])[0]+1):
            if i!=expId:
                temp.append(i)
                
        inds.append(temp)    
        f1_scores.append(tempF1_score)
        realF1_scores.append(tempF1_scoreReal)
    
    #Ranking
    for i in range(len(f1_scores)):
        tempRanks=ranker(f1_scores[i])
        temp=[inds[i][i2] for i2 in tempRanks]
        
        tempRealRanks=ranker(realF1_scores[i])
        tempReal=[inds[i][i2] for i2 in tempRealRanks]
        
        realRanks.append(tempReal)
        ranks.append(temp)
        
        #Spearman's ranking calculation
        mrealRank=range(len(tempReal))
        tempReal=np.array(tempReal)
        mrank=[int(np.argwhere(tempReal==i).ravel()) for i in temp]
        mrank=np.array(mrank)+1
        mrealRank=np.array(mrealRank)+1
        spears.append(spearmanr(mrank,mrealRank))
    
    return ranks,realRanks,spears,inds

def monotonicRanker(rank1,rank2):
    '''
    This function takes an index based rank and then transforms it in to a 
    monotonic rank. For example: If we have rank1=[3,2,1] which corresponds to model 3 is at the top, 2 is 
    the second and so on. And we have rank2=[3,1,2] Then the monotic ranks correspond to
    mRank1=[1,2,3] and mRank2=[1,3,2]
    '''
    rank1=np.array(rank1)
    rank2=np.array(rank2)
    mRank1=range(1,len(rank1)+1)
    mRank2=[ int(np.argwhere(rank2==i))+1 for i in rank1]
#     print(mRank1,mRank2)
    return mRank1,mRank2

def agreementRates(clf,valLabels,uLabels,plot=False):
    '''
    This function gets a classifiers dictionary generated from clfsEval
    and calculates the agreement of the classifiers and
    generate a list of accuracy, f1_scores and other performance metrics
    so that we can figure out the thold that can maximize performance
    '''
    
    clfsLabels=[i for i in clf['pred_val']]
    clfsLabels=np.array(clfsLabels)
    
    agmnt=[]
    mvLabels=[]
    
    #Calculation of the majority vote labels
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
            mvLabels.append(cts['vals'][tind])
        else:
            tind=np.random.permutation(tind)[0]
            mvLabels.append(cts['vals'][tind])
            
        
    sAgmnt=np.sort(np.unique(agmnt))
    clf['agmntLevels']=sAgmnt
    
    #In this section is calculated the f1_score for the ground truth and
    #the predicted values filtering by the agreement rate
    f1_scores=[]
    for i in sAgmnt:
        inds2=np.argwhere(agmnt>=i)
        temp=[]
        for cnum in range(len(clf['pred_val'])):
            try:
                temp.append(f1_score(valLabels[inds2],clf['pred_val'][cnum][inds2],labels=uLabels))
            except:
                print('problem at funcs agreement rates')
#             print(f1_score(valLabels[inds2],clf['pred_val'][cnum][inds2],labels=uLabels))
        f1_scores.append(temp)
    if plot==True:
        plt.plot(sAgmnt,f1_scores)
        plt.show()
    clf['f1_score_val_predval_agmnt']=f1_scores
    
    return mvLabels,agmnt
    
        
def clfsVal(clfs,mvLabels,agmntLevels,uLabels):
    '''
    Take the classifiers and the already calculated outputs and measure their f1_score
    for the labels generated from the majority vote
    This function does not return anything but instead adds an entry to the
    dictionary
    This function can actually be performed inside agreement rates
    '''
    clfs['f1_score_mv_predval_agmnt']=[]
    
    #Here is calculated the f1_score for the majority vote labels and the predicted 
    #values filtering by the agreement level    
    agmntLvls=agmntLevels
    for agmntLvl in clfs['agmntLevels']:
        temp=[]
        for i in range(len(clfs['pred_val'])):
            inds=np.argwhere(np.array(agmntLevels)>=agmntLvl)
            agmntF1=f1_score(np.array(mvLabels)[inds],np.array(clfs['pred_val'][i])[inds],labels=uLabels)
            temp.append(agmntF1)
        clfs['f1_score_mv_predval_agmnt'].append(temp)
        
    
def ranker(ranks):
    '''
    This function gives back the indices of ranks
    organized from highest to lowest values. It also
    takes into account when there are similar values
    and decides by chance in those cases. Though it only
    works for ties at the top level. ranks are the performance scores from 
    which the ranks are going to be drawn
    ''' 
       
    inds=np.argsort(ranks)[::-1]
    id=[]
    output=[]
    
    for i in range(0,len(inds)):
        if ranks[inds[0]]==ranks[inds[i]]:
            id.append(inds[i])
    
    if len(id)>1:    
        id=np.random.choice(id,len(id),replace=False)
        return np.hstack((id,inds[len(id):]))
    else:
        return inds
    
def spearmansCalc(clfs):
    rVal=ranker(clfs['f1_score_val_predval'])
    clfs['rank_val_predval']=rVal
    clfs['rank_mv_predval_agmnt']=[]
    
    rAgmnt=[]
    clfs['spearmans']=[]
    for i in range(len(clfs['agmntLevels'])):
        rAgmnt=ranker(clfs['f1_score_mv_predval_agmnt'][i])
        clfs['rank_mv_predval_agmnt'].append(rAgmnt)
        rankVal=range(len(rVal))
        rankAgmnt=[int(np.argwhere(rVal==i)) for i in rAgmnt]
        rankVal=np.array(rankVal)+1
        rankAgmnt=np.array(rankAgmnt)+1
        
        clfs['spearmans'].append(spearmanr(rankVal,rankAgmnt))
    

def summaryGraphs(f1_score_mv_predval_agmnt,classifiersNum,spear,ranks,bestF1s,agmntLevels):
    bestF1s=np.array(bestF1s)
    data=[]
    tData=[]
    spears=[]
    plt.hold(True)
    
    
    
    for i in range(len(spear)):
        if spears==[]:
            try:
                spears=np.array(spear[i])[:,0]
            except:
                print('problem')
        else:
            try:
                spears=np.vstack((spears,(np.array(spear[i])[:,0])))
            except:
                print('problem')
    
    for i2 in range(classifiersNum):
        data=[]
        #Going through each of the time series data points
        for i in range(len(f1_score_mv_predval_agmnt)):
            data.append(np.array(f1_score_mv_predval_agmnt[i])[:,i2])
        data=np.transpose(np.array(data))
        if tData==[]:
            tData=data
        else:
            tData=np.vstack((tData,data))
        plt.errorbar(range(1,1+len(f1_score_mv_predval_agmnt)),np.mean(data,0),yerr=np.std(data,0),label='clf %d'%(i2))
    plt.errorbar(range(1,1+len(f1_score_mv_predval_agmnt)),np.mean(spears,0),yerr=np.std(spears,0),label="spearman's")
#     plt.boxplot(tData)
    plt.errorbar(range(1,1+np.shape(bestF1s)[1]),np.mean(bestF1s,0),yerr=np.std(bestF1s,0),label='f1_score best model')
    plt.legend(loc=3)
#     ax=plt.gca()
#     labels = [item.get_text() for item in ax.get_xticklabels()]
#     ax.set_xticklabels([str(i) for i in agmntLevels])
    plt.xticks(range(1,1+np.shape(bestF1s)[1]),[str(i) for i in agmntLevels])
    plt.show()
    
    #Now I have to add to the code a way to plot the mode of the ranks for each agreement level, the
    #data should be on ranks already
    
    
if __name__=='__main__':
#     temp=[5,5,5,4,7,7,7,1,3,2]
#     print(np.argsort(temp)[::-1])
#     print(ranker(temp))
    temp=[3,2,1]
    temp2=[3,1,2]
    
    print(monotonicRanker(temp,temp2))
    
        
        
        
    
    
    

        

    
    
    
    
