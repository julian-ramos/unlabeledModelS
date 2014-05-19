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

filename=intermediatePath+'/'+'summary.plk'
file=open(filename,'rb')
summary=pickle.load(file)
print('summary data loaded')
agmnt=summary['agmnt']
mv=summary['mv']
votes=summary['votes']

ranks=fun.ranking(agmnt,mv,votes)

#Now that the ranks are ready I can compare them with the ranks
#I get using the real labels from the data
    

