from numpy import loadtxt
from xgboost import XGBClassifier
from catboost import CatBoostClassifier 
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn import metrics
from xgboost import plot_importance
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

# init arrays for saving test scores
meanAUC = np.array([])
meanPrecision = np.array([])
meanRecall = np.array([])
meanAccuracy = np.array([])
meanF1score = np.array([])

# init arrays for saving train scores
TRAINmeanAUC = np.array([])
TRAINmeanPrecision = np.array([])
TRAINmeanRecall = np.array([])
TRAINmeanAccuracy = np.array([])
TRAINmeanF1score = np.array([])

# init arrays for saving val scores
VALmeanAUC = np.array([])
VALmeanPrecision = np.array([])
VALmeanRecall = np.array([])
VALmeanAccuracy = np.array([])
VALmeanF1score = np.array([])

# init a list for feature selection (we have 17 features)
feature_vote = [0 for x in range(17)]

# we produce the 100 datasets by random resampling from the original dataset
for i in range(0, 100):
    print('DataSet%d -------------------' % (i + 1))

    # load data
    df = pd.read_csv("tmpData100/tmp" + str(i + 1) + ".csv")
    dataset = np.array(df)
    
    # choose 35 samples from whole 53 samples for training
    X_train = dataset[:35,:-1]

    # add the smote sample to the unbalanced dataset
    X_train = np.vstack((X_train, X_samp))

    # set the test dataset
    X_test = dataset[35:,:-1]

    # select features
    VInd=[i for i in range(0,17)]
    X_train = X_train[:, VInd]
    X_test = X_test[:, VInd]
    y_train = dataset[:35, -1]
    y_train = np.append(y_train, y_samp)
    y_test = dataset[35:, -1]
    
    # choose different models to train the classification models

    # K-fold cross validation
    K = 7
    head_ct = len(X_train)
    each_ct = int(head_ct / K)
    # save train/test/val scores
    dict_score = {'test_score': [], 'train_score': [], 'val_score': []}
    for k in range(K):
        X_train_k = np.delete(X_train, range(k * each_ct, (k + 1) * each_ct), axis=0)
        y_train_k = np.delete(y_train, range(k * each_ct, (k + 1) * each_ct))
        # init different models
        model = XGBClassifier(max_depth=6, subsample=1.0, min_child_weight=1.0, gamma=0.3,
            learning_rate=0.2, n_estimators=100, colsample_bytree=0.25, eval_metric='auc')
        model.fit(X_train_k, y_train_k)
        
        # caculate the train scores
        y_pred_train = model.predict(X_train_k)
        fpr, tpr, thresholds = metrics.roc_curve(y_train_k, y_pred_train)
        roc_auc = metrics.auc(fpr, tpr)
        precision = metrics.precision_score(y_train_k, y_pred_train)
        recall = metrics.recall_score(y_train_k, y_pred_train)
        accuracy = metrics.accuracy_score(y_train_k, y_pred_train)
        f1_score = metrics.f1_score(y_train_k, y_pred_train)
        dict_score['train_score'].append([roc_auc, precision, recall, accuracy, f1_score])
        
        # caculate the val scores
        x_val_k = X_train[k * each_ct:(k + 1) * each_ct,:]
        y_val_k = y_train[k * each_ct:(k + 1) * each_ct]
        y_pred_val = model.predict(x_val_k)
        fpr, tpr, thresholds = metrics.roc_curve(y_val_k, y_pred_val)
        roc_auc = metrics.auc(fpr, tpr)
        precision = metrics.precision_score(y_val_k, y_pred_val)
        recall = metrics.recall_score(y_val_k, y_pred_val)
        accuracy = metrics.accuracy_score(y_val_k, y_pred_val)
        f1_score = metrics.f1_score(y_val_k, y_pred_val)
        dict_score['val_score'].append([roc_auc, precision, recall, accuracy, f1_score])

        # caculate the test scores
        y_pred = model.predict(X_test)
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
    
    # metrics caculator (train)
    all_scores_train = np.mean(np.array(dict_score['train_score']), axis=0)
    [roc_auc, precision, recall, accuracy, f1_score] = all_scores_train
    print("train roc_auc_score = %0.4f " % (roc_auc))
    print("train Precision =", precision)
    print("train Recall =", recall)
    print("train Accuracy =", accuracy)
    print("train F1score =", f1_score)
    TRAINmeanAUC = np.append(TRAINmeanAUC, roc_auc)
    TRAINmeanPrecision = np.append(TRAINmeanPrecision, precision)
    TRAINmeanRecall = np.append(TRAINmeanRecall, recall)
    TRAINmeanAccuracy = np.append(TRAINmeanAccuracy, accuracy)
    TRAINmeanF1score = np.append(TRAINmeanF1score, f1_score)
    
    # metrics caculator (val)
    all_scores_val = np.mean(np.array(dict_score['val_score']), axis=0)
    [roc_auc, precision, recall, accuracy, f1_score] = all_scores_val
    print("val roc_auc_score = %0.4f " % (roc_auc))
    print("val Precision =", precision)
    print("val Recall =", recall)
    print("val Accuracy =", accuracy)
    print("val F1score =", f1_score)
    VALmeanAUC = np.append(VALmeanAUC, roc_auc)
    VALmeanPrecision = np.append(VALmeanPrecision, precision)
    VALmeanRecall = np.append(VALmeanRecall, recall)
    VALmeanAccuracy = np.append(VALmeanAccuracy, accuracy)
    VALmeanF1score = np.append(VALmeanF1score, f1_score)

# select features bases on different models
vi_ind = list(map(feature_vote.index, heapq.nlargest(10, feature_vote)))
print(vi_ind)

# save the scores to csv files
dataFrame = pd.DataFrame({'AUC': meanAUC, 'Precision': meanPrecision, 'Recall': meanRecall, 'Accuracy': meanAccuracy, 'F1_score': meanF1score,
    'trainAUC': TRAINmeanAUC, 'trainPrecision': TRAINmeanPrecision, 'trainRecall': TRAINmeanRecall, 'trainAccuracy': TRAINmeanAccuracy,
    'trainF1score': TRAINmeanF1score})
dataFrame.to_csv('score/smote/xgboost.csv', index=True, sep=',')

print('------------------test scores------------------')
print('test meanAUC= %0.4f' % (np.mean(meanAUC)))
print('test meanPrecision= %0.4f' % (np.mean(meanPrecision)))
print('test meanRecall= %0.4f' % (np.mean(meanRecall)))
print('test meanAccuracy= %0.4f' % (np.mean(meanAccuracy)))
print('test meanF1score= %0.4f' % (np.mean(meanF1score)))

print('------------------train scores------------------')
print('train meanAUC= %0.4f' % (np.mean(TRAINmeanAUC)))
print('train meanPrecision= %0.4f' % (np.mean(TRAINmeanPrecision)))
print('train meanRecall= %0.4f' % (np.mean(TRAINmeanRecall)))
print('train meanAccuracy= %0.4f' % (np.mean(TRAINmeanAccuracy)))
print('train meanF1score= %0.4f' % (np.mean(TRAINmeanF1score)))

print('------------------val scores------------------')
print('val meanAUC= %0.4f' % (np.mean(VALmeanAUC)))
print('val meanPrecision= %0.4f' % (np.mean(VALmeanPrecision)))
print('val meanRecall= %0.4f' % (np.mean(VALmeanRecall)))
print('val meanAccuracy= %0.4f' % (np.mean(VALmeanAccuracy)))
print('val meanF1score= %0.4f' % (np.mean(VALmeanF1score)))
