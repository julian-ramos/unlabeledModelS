#===============================================================================
#THis unit trains a defined Machine learning classification model
#per user and stores performance metrics as well as majority votes 
#for a leave one user out scheme
# Copyright (C) Julian Ramos 2014
#===============================================================================
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
clfs=[]
# This options tell the program to split the data into testing,training and validation data 
#sets 
testAndVal=False
# This option cuts out the quantity of classifiers to only 3 for fast evaluation
fast=False
cont=0
if testAndVal:
    print('training,testing and validation data sets are going to be created')
else:
    print('training using the entire data set')


allData={'user':[],'trainData':[],'trainLabels':[],\
         'testData':[],'testLabels':[],\
         'valData':[],'valLabels':[],'features':[]\
         }




#Check if models already exist
if os.listdir(modelsPath)!=[]:
    print('Models already created, loading them')
    print('loading models ...')
    filename=modelsPath+'/'+os.listdir(modelsPath)[0]
    f=open(filename,'rb')
    clfs=pickle.load(f)
#     print('loading data ...')
#     filename=intermediatePath+'/'+os.listdir(intermediatePath)[0]
#     f=open(filename,'rb')
#     allData=pickle.load(f)
    
else:
    print('Models not found')
    print('building models ...')
    #Creating the training models for each user
    for file in filesList:
        #Data load
        filename=dataPath+'/'+file
        data,labels,features=fun.dataExtract(filename)
        print('training model for file %s'%(file))
        if testAndVal==True:
            #This section if we want to have data split into training, testing and validation
    #         Data separation
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
            allData['user'].append(file)
            allData['trainData'].append(trainData)
            allData['trainLabels'].append(trainLabels)
            allData['testData'].append(testData)
            allData['testLabels'].append(testLabels)
            allData['valData'].append(valData)
            allData['valLabels'].append(valLabels)
            allData['features'].append(features)
            
        else:
            #The data is not splitted
            uLabels=np.unique(labels)
            tempClf=fun.clfsEval(data,data[:100,],data[100:200,:],labels,labels[:100],labels[100:200],uLabels,classN=1)
            #There's no need to store the data
        clfs.append(tempClf)
        cont+=1
        if cont==6 and fast==True:
            break
        

    # Storing all the classifiers and results
    print('Storing the models')    
    filename=modelsPath+'/'+'models.plk'
    f=open(filename,'wb')
    pickle.dump(clfs,f)
    f.close()
    if testAndVal==True:
        # Storing the data
        print('Storing the data')
        filename=intermediatePath+'/'+'data.plk'
        f=open(filename,'wb')
        pickle.dump(allData,f)
        f.close()
 
# mvAll=[]
# #Calculating the majority vote for each of the data sets
# 
# 
# for uId in range(len(allData)):
#     for cId in range(len(clfs)):
#         if uId!=cId:
#             mvAll.append(clfs[cId]['classifier'].predict(allData['trainData'][uId]))
# 

print('computing agreement levels...')
summary={'mv':[],'agmnt':[],'user':[],'votes':[],'labels':[]}
#Curently simply building the models and storing afterwards I have to compute the 
#agreement rate
if testAndVal:
    print('not implemented')
else:
    for fInd in range(len(filesList)):
        file=filesList[fInd]
        filename=dataPath+'/'+file
        data,labels,features=fun.dataExtract(filename)
        tempMv,tempAgmnt,tempVotes=fun.majorityVote(clfs,data,[fInd])
        summary['mv'].append(tempMv)
        summary['agmnt'].append(tempAgmnt)
        summary['user'].append(file)
        summary['votes'].append(tempVotes)
        summary['labels'].append(labels)

print('storing summary')
filename=intermediatePath+'/'+'summary.plk'
file=open(filename,'wb')
pickle.dump(summary,file)
file.close()
print('summary stored at %s'%(filename))
        
        

