import matplotlib.pylab as plt
from sklearn.metrics import f1_score
import dataSearchUtils as dSu
import funcs as fun
import dataReader as dR
import os
import dataML_Prepro as dmlPre
import numpy as np
import pickle


intermediatePath='F:/Activity_RS/Last data set/FeatureDataset/intermediateData'
modelsPath='F:/Activity_RS/Last data set/FeatureDataset/models'
dataPath='F:/Activity_RS/Last data set/FeatureDataset/modelSelectionProjectData'
filesList=os.listdir(dataPath)
spearsa=[]
spearsIS=[]

filename=intermediatePath+'/'+'summary.plk'
file=open(filename,'rb')
summary=pickle.load(file)
print('summary data loaded')
agmnt=summary['agmnt']
mv=summary['mv']
votes=summary['votes']
labels=summary['labels']

agmntLvls=[np.unique(i).tolist() for i in agmnt]

agmntLvls=[ np.min(i) for i in agmntLvls]
agmntLvls=np.arange(np.max(agmntLvls),1,0.1)

# agmntLvls=[ np.max(i) for i in agmntLvls]
# agmntLvls=np.arange(np.max(agmntLvls),0.9,0.1)


for agmntVal in agmntLvls: 
    ranksMv,realRanks,spears,inds=fun.ranking(agmnt,mv,votes,labels,agmntLvl=agmntVal)
    #I have to elimitante this inds because it is going to give trouble later
    #inds should be part of the data extracted from the summary file
    score,ranksIS,spearsis=fun.izbickiSternScore(votes,inds,realRanks)
    spearsIS.append(np.array(spearsis)[:,0])
    spearsa.append(np.array(spears)[:,0])
    print(np.mean(np.array(spears),0))
    print(np.median(np.array(spears),0))
    print(np.std(np.array(spears),0))
    
plt.hold(True)
s1=plt.boxplot(spearsa)
s2=plt.boxplot(spearsIS)

plt.setp(s1['boxes'], color='black')
plt.setp(s1['whiskers'], color='black')
plt.setp(s1['fliers'], color='black', marker='+')
plt.setp(s1['medians'], color='black')

plt.setp(s2['boxes'], color='red')
plt.setp(s2['whiskers'], color='red')
plt.setp(s2['fliers'], color='red', marker='+')
plt.setp(s2['medians'], color='red')
#     plt.legend(s,['%d'%(agmntVal)])
plt.show()

# Currently there is a problem with the ranking procedure which is 
# that I'm solving the ties by drawing at random however I shouldn't do that
# instead the mean rank across the ones that are tied is calculated
# I can probably solve this issue by extending the ranker function
# to output two values instead of one. The second value is the rank while
# the first value is the model number itself ordered by rank
# This I believe penalizes less harshly the ranking for ties


print('done')

#Now that the ranks are ready I can compare them with the ranks
#I get using the real labels from the data
    

