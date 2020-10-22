from numpy import loadtxt
from xgboost import XGBClassifier
from catboost import CatBoostClassifier 
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn import metrics
import smote_variants as sv
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import heapq

# prepare the smote data --> X_samp, y_samp
df = pd.read_csv("original_data.csv")
data = np.array(df)
X = data[:,:-1]
y = data[:, -1]
oversampler = sv.distance_SMOTE()
X_samp, y_samp = oversampler.sample(X, y)
X_samp = np.round(X_samp)
y_samp = np.round(y_samp)
X_samp, y_samp = X_samp[len(X):], y_samp[len(y):]
# set the count of smote data
SmoteNum = 11
X_samp = X_samp[:SmoteNum,:]
y_samp = y_samp[:SmoteNum]

# init arrays for saving test scores and train scores
meanAUC = np.array([])
meanPrecision = np.array([])
meanRecall = np.array([])
meanAccuracy = np.array([])
meanF1score = np.array([])

TRAINmeanAUC = np.array([])
TRAINmeanPrecision = np.array([])
TRAINmeanRecall = np.array([])
TRAINmeanAccuracy = np.array([])
TRAINmeanF1score = np.array([])

# init a list for feature selection (we have 17 features)
feature_vote = [0 for x in range(17)]

# we produce the 100 datasets by random resampling from the original dataset
for i in range(0, 100):
    print('DataSet%d -------------------' % (i + 1))
    
    # load data
    df = pd.read_csv("tmpData100/tmp" + str(i + 1) + ".csv")
    dataset = np.array(df)
    
    # use different models to train the ensemble classification models
    # save prediction results from different models
    y_pred_list = []
    y_train_pred_list=[]

    # XGBoost
    model = XGBClassifier(max_depth=6, subsample=1.0, min_child_weight=1.0, gamma=0.3,
        learning_rate=0.2, n_estimators=100, colsample_bytree=0.25, eval_metric='auc')
    X_train = dataset[:35,:-1]
    X_train = np.vstack((X_train, X_samp))
    X_test = dataset[35:,:-1]
    VInd = [i for i in range(0, 17)]
    X_train = X_train[:, VInd]
    X_test = X_test[:, VInd]
    y_train = dataset[:35, -1]
    y_train = np.append(y_train, y_samp)
    y_test = dataset[35:, -1]
    model.fit(X_train, y_train, verbose=False)
    y_pred_list.append(model.predict(X_test))
    y_train_pred_list.append( model.predict(X_train) )

    # logistic
    model = LogisticRegression(max_iter=300, C=1.0)
    X_train = dataset[:35,:-1]
    X_train = np.vstack((X_train, X_samp))
    X_test = dataset[35:,:-1]
    VInd = [0, 3, 4, 7, 10, 13, 14, 16]
    X_train = X_train[:, VInd]
    X_test = X_test[:, VInd]
    y_train = dataset[:35, -1]
    y_train = np.append(y_train, y_samp)
    y_test = dataset[35:, -1]
    model.fit(X_train, y_train)
    y_pred_list.append(model.predict(X_test))
    y_train_pred_list.append(model.predict(X_train))

    # SVM
    model = svm.SVC(kernel='linear', C=0.3)
    X_train = dataset[:35,:-1]
    X_train = np.vstack((X_train, X_samp))
    X_test = dataset[35:,:-1]
    VInd = [4, 3, 13, 10, 5, 12, 15, 16]
    X_train = X_train[:, VInd]
    X_test = X_test[:, VInd]
    y_train = dataset[:35, -1]
    y_train = np.append(y_train, y_samp)
    y_test = dataset[35:, -1]
    model.fit(X_train, y_train)
    y_pred_list.append(model.predict(X_test))
    y_train_pred_list.append( model.predict(X_train) )

    # CatBoost
    model = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", learning_rate=0.1,
        depth=6, iterations=100, border_count=20, subsample=1.0, colsample_bylevel=1.0,
        random_strength=0.7, scale_pos_weight=1, reg_lambda=10)
    X_train = dataset[:35,:-1]
    X_train = np.vstack((X_train, X_samp))
    X_test = dataset[35:,:-1]
    VInd = [i for i in range(0, 17)]
    X_train = X_train[:, VInd]
    X_test = X_test[:, VInd]
    y_train = dataset[:35, -1]
    y_train = np.append(y_train, y_samp)
    y_test = dataset[35:, -1]
    model.fit(X_train, y_train, verbose=False)
    y_pred_list.append(model.predict(X_test))
    y_train_pred_list.append( model.predict(X_train) )

    # get the mean prediction results from different models
    y_pred = np.round(np.mean(np.array(y_pred_list), axis=0)).astype('int')
    y_pred_train = np.round( np.mean( np.array(y_train_pred_list),axis=0)).astype('int')

    # set a dict for saving scores
    dict_score = {'train_score': [], 'test_score': []}

    # caculate score for testing
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)
    dict_score['test_score'].append([roc_auc, precision, recall, accuracy, f1_score])

    # metrics caculator (test)
    all_scores_test = np.mean(np.array(dict_score['test_score']), axis=0)
    [roc_auc, precision, recall, accuracy, f1_score] = all_scores_test
    print("test roc_auc_score = %0.4f " % (roc_auc))
    print("test Precision =", precision)
    print("test Recall =", recall)
    print("test Accuracy =", accuracy)
    print("test f1_score =", f1_score)
    meanAUC = np.append(meanAUC, roc_auc)
    meanPrecision = np.append(meanPrecision, precision)
    meanRecall = np.append(meanRecall, recall)
    meanAccuracy = np.append(meanAccuracy, accuracy)
    meanF1score = np.append(meanF1score, f1_score)

    # caculate score for training
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred_train)
    roc_auc = metrics.auc(fpr, tpr)
    precision = metrics.precision_score(y_train, y_pred_train)
    recall = metrics.recall_score(y_train, y_pred_train)
    accuracy = metrics.accuracy_score(y_train, y_pred_train)
    f1_score = metrics.f1_score(y_train, y_pred_train)
    dict_score['train_score'].append([roc_auc, precision, recall, accuracy, f1_score])
    
    # metrics caculator (train)
    all_scores_train = np.mean(np.array(dict_score['train_score']), axis=0)
    [roc_auc, precision, recall, accuracy, f1_score] = all_scores_train
    TRAINmeanAUC = np.append(TRAINmeanAUC, roc_auc)
    TRAINmeanPrecision = np.append(TRAINmeanPrecision, precision)
    TRAINmeanRecall = np.append(TRAINmeanRecall, recall)
    TRAINmeanAccuracy = np.append(TRAINmeanAccuracy, accuracy)
    TRAINmeanF1score = np.append(TRAINmeanF1score, f1_score)

dataFrame=pd.DataFrame({'AUC':meanAUC,'Precision':meanPrecision,'Recall':meanRecall,'Accuracy':meanAccuracy,'F1_score':meanF1score,
    'trainAUC':TRAINmeanAUC,'trainPrecision':TRAINmeanPrecision,'trainRecall':TRAINmeanRecall,'trainAccuracy':TRAINmeanAccuracy,
    'trainF1score': TRAINmeanF1score})
dataFrame.to_csv("score/smote/ensemble_learning_results.csv", index=True, sep=',')

print('------------------test scores------------------')
print('test meanAUC= %0.4f' % (np.mean(meanAUC)))
print('test meanPrecision= %0.4f' % (np.mean(meanPrecision)))
print('test meanRecall= %0.4f' % (np.mean(meanRecall)))
print('test meanAccuracy= %0.4f' % (np.mean(meanAccuracy)))
print('test meanF1score= %0.4f' % (np.mean(meanF1score)))
print('------------------train scores------------------')
print('train meanAUC= %0.4f'% ( np.mean(TRAINmeanAUC)))
print('train meanPrecision= %0.4f'% ( np.mean(TRAINmeanPrecision)))
print('train meanRecall= %0.4f'% ( np.mean(TRAINmeanRecall)))
print('train meanAccuracy= %0.4f'% ( np.mean(TRAINmeanAccuracy)))
print('train meanF1score= %0.4f'% ( np.mean(TRAINmeanF1score)))