from numpy import loadtxt
from xgboost import XGBClassifier
from catboost import CatBoostClassifier 
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from xgboost import plot_importance
import smote_variants as sv
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import imbalanced_databases as imbd
import sklearn.datasets as datasets
import os.path

#prepare smote sample1 31  --> X_samp, y_samp

df = pd.read_csv("dev.csv")
data =np.array(df)
X=data[:,:-1]
y=data[:,-1]
oversampler= sv.distance_SMOTE()
X_samp, y_samp= oversampler.sample(X, y)
X_samp=np.round(X_samp)
y_samp=np.round(y_samp)
X_samp, y_samp= X_samp[len(X):], y_samp[len(y):]
SmoteNum=20
X_samp=X_samp[:SmoteNum,:]
y_samp= y_samp[:SmoteNum]


#prepare smote2 sample 


#--------------------------------------------XGBoost 
meanAUC=np.array([])
meanPrecision=np.array([])
meanRecall=np.array([])
meanAccuracy=np.array([])
TRAINmeanAUC=np.array([])
TRAINmeanPrecision=np.array([])
TRAINmeanRecall=np.array([])
TRAINmeanAccuracy=np.array([])
for i in range(0,10):
    print('DataSet%d -------------------'%(i+1))
    # load data
    df = pd.read_csv("../tmpData10/tmp"+str(i+1)+".csv")
    dataset =np.array(df)
    X_train=dataset[:35,:-1]
    X_train=np.vstack((X_train, X_samp ))

    X_test= dataset[35:,:-1]
    #choose some features
    #VInd=[4,5,3,1,13]
    VInd=[i for i in range(0,17)]
    X_train=X_train[:,VInd]
    X_test=X_test[:,VInd]

    y_train=dataset[:35,-1]
    y_train=np.append( y_train, y_samp )

    y_test= dataset[35:,-1]
    #fit model no training data
    '''
    #best paras for 17 features XGBoost
    model = XGBClassifier(max_depth=6,subsample=1.0,min_child_weight=1.0,gamma=0.3,
        learning_rate=0.2,n_estimators=100,colsample_bytree=0.25,eval_metric='auc')
    '''
    # best paras for smote data
    model = XGBClassifier(max_depth=6,subsample=1.0,min_child_weight=1.0,gamma=0.3,
        learning_rate=0.2,n_estimators=100,colsample_bytree=0.25,eval_metric='auc')

    '''
    #best paras for VInd=[4,5,3,1,13] XGBoost
    model = XGBClassifier(max_depth=6,subsample=0.9,min_child_weight=1.0,gamma=0.3,
    learning_rate=0.2,n_estimators=100,colsample_bytree=0.25,eval_metric='auc')
    '''
    
    model.fit(X_train , y_train, verbose=True)
    # make predictions for test data
    y_pred = model.predict(X_test )
    y_pred_train=model.predict(X_train)
    # metrics caculator (test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test , y_pred)
    roc_auc=metrics.auc(fpr,tpr)
    print("test roc_auc_score = %0.4f "% (roc_auc))
    print("test Precision =", metrics.precision_score(y_test, y_pred))
    print("test Recall =", metrics.recall_score(y_test, y_pred))
    print("test Accuracy =", metrics.accuracy_score(y_test, y_pred))
    meanAUC=np.append(meanAUC,roc_auc)
    meanPrecision=np.append(meanPrecision,metrics.precision_score(y_test, y_pred))
    meanRecall=np.append(meanRecall,metrics.recall_score(y_test, y_pred))
    meanAccuracy=np.append(meanAccuracy, metrics.accuracy_score(y_test, y_pred))

    # metrics caculator (train)
    fpr, tpr, thresholds = metrics.roc_curve(y_train , y_pred_train)
    roc_auc=metrics.auc(fpr,tpr)
    print("train roc_auc_score = %0.4f "% (roc_auc))
    print("train Precision =", metrics.precision_score(y_train , y_pred_train))
    print("train Recall =", metrics.recall_score(y_train , y_pred_train))
    print("train Accuracy =", metrics.accuracy_score(y_train , y_pred_train))
    TRAINmeanAUC=np.append(TRAINmeanAUC,roc_auc)
    TRAINmeanPrecision=np.append(TRAINmeanPrecision, metrics.precision_score(y_train , y_pred_train))
    TRAINmeanRecall=np.append(TRAINmeanRecall,metrics.recall_score(y_train , y_pred_train))
    TRAINmeanAccuracy=np.append(TRAINmeanAccuracy, metrics.accuracy_score(y_train , y_pred_train))
'''
dataFrame=pd.DataFrame({'AUC':meanAUC,'Precision':meanPrecision,'Recall':meanRecall,'Accuracy':meanAccuracy,
    'trainAUC':TRAINmeanAUC,'trainPrecision':TRAINmeanPrecision,'trainRecall':TRAINmeanRecall,'trainAccuracy':TRAINmeanAccuracy})
dataFrame.to_csv("scores.csv",index=True,sep=',')
'''
print('------------------分界线')
print('test meanAUC= %0.4f'% ( np.mean(meanAUC)))
print('test meanPrecision= %0.4f'% ( np.mean(meanPrecision)))
print('test meanRecall= %0.4f'% ( np.mean(meanRecall)))
print('test meanAccuracy= %0.4f'% ( np.mean(meanAccuracy)))
print('train meanAUC= %0.4f'% ( np.mean(TRAINmeanAUC)))
print('train meanPrecision= %0.4f'% ( np.mean(TRAINmeanPrecision)))
print('train meanRecall= %0.4f'% ( np.mean(TRAINmeanRecall)))
print('train meanAccuracy= %0.4f'% ( np.mean(TRAINmeanAccuracy)))

'''
#draw feature importance
df = pd.read_csv("dev.csv")
dataset =np.array(df)
X_train=dataset[:,:-1]
y_train=dataset[:,-1]
#fit model no training data
model = XGBClassifier(max_depth=6,subsample=1.0,min_child_weight=1.0,gamma=0.3,
    learning_rate=0.2,n_estimators=100,colsample_bytree=0.25,eval_metric='auc')
model.fit(X_train , y_train, verbose=True)
plot_importance(model)
plt.show()
'''