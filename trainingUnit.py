#===============================================================================
#THis unit trains a defined Machine learning classification model
#per user and stores performance metrics as well as majority votes 
#for a leave one user out scheme
# Copyright (C) Julian Ramos 2014
#===============================================================================

import dataSearchUtils as dSu
import funcs as fun
import dataReader as dR
import os
import dataML_Prepro as dmlPre
import numpy as np
import pickle

modelsPath='F:/Activity_RS/Last data set/FeatureDataset/models'
dataPath='F:/Activity_RS/Last data set/FeatureDataset/modelSelectionProjectData'
filesList=os.listdir(dataPath)
clfs=[]

#Creating the training models for each user
for file in filesList:
    #Data load
    filename=dataPath+'/'+file
    data,labels=fun.dataExtract(filename)
    
    #Data separation
    splitData=dmlPre.dataSplitBalancedClass(data, labels)
    trainData=splitData['trainData']
    trainLabels=splitData['trainLabels']
    testData=splitData['testData']
    testLabels=splitData['testLabels']
    valData=splitData['valData']
    valLabels=splitData['valLabels'] 
    uLabels=np.unique(np.hstack((np.unique(trainLabels),np.unique(testLabels),np.unique(valLabels))))
    
    #Train the different classifiers and predict for the testing and validation data sets
    tempClf=fun.clfsEval(trainData,testData,valData,trainLabels,testLabels,valLabels,uLabels,classN=1)
    clfs.append(tempClf)

# Storing all the classifiers and results    
filename=modelsPath+'/'+'models.plk'
f=open(filename,'wb')
pickle.dump(clfs,f)

# Currently everything seems fine, 
# the only big big problem is that 
# the results that I'm getting are too good to be true'

print('here')

