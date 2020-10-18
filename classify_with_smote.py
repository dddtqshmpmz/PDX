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
    VInd = [i for i in range(0, 17)]
    X_train = X_train[:, VInd]
    X_test = X_test[:, VInd]
    y_train = dataset[:35, -1]
    y_train = np.append(y_train, y_samp)
    y_test = dataset[35:, -1]
    
    # choose different models to train the classification models
    # XGBoost
    model = XGBClassifier(max_depth=6, subsample=1.0, min_child_weight=1.0, gamma=0.3,
        learning_rate = 0.2, n_estimators = 100, colsample_bytree = 0.25, eval_metric = 'auc') 
        
    # Logistic model
    model = LogisticRegression(max_iter=300, C=1.0)
    
    # SVM model
    model = svm.SVC(kernel='linear', C=0.3)
    
    # CatBoost
    model = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", learning_rate=0.1,
        depth=6, iterations=100, border_count=20, subsample=1.0, colsample_bylevel=1.0,
        random_strength=0.7, scale_pos_weight=1, reg_lambda=10)
    
    model.fit(X_train, y_train)

    # select the most important features according to the model.feature_importances_ or the model.coef_
    # for svm and logistic models
    feaImportances = list(model.coef_[0])
    # for xgboost and catboost models
    feaImportances = list(model.feature_importances_) 
    feaImportances = [abs(x) for x in feaImportances]

    vi_ind = list(map(feaImportances.index, heapq.nlargest(17, feaImportances)))
    print(vi_ind)
    for rank in range(10):
        feature_vote[vi_ind[rank]] += feaImportances[vi_ind[rank]]

    # make predictions for test data
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # metrics caculator (test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)
    print("test roc_auc_score = %0.4f " % (roc_auc))
    print("test Precision =", precision)
    print("test Recall =", recall)
    print("test Accuracy =", accuracy)
    print("test f1_score =", f1_score)

    # save the scores in arrays (test)
    meanAUC = np.append(meanAUC, roc_auc)
    meanPrecision = np.append(meanPrecision, precision)
    meanRecall = np.append(meanRecall, recall)
    meanAccuracy = np.append(meanAccuracy, accuracy)
    meanF1score = np.append(meanF1score, f1_score)
    
    # metrics caculator (train)
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred_train)
    roc_auc = metrics.auc(fpr, tpr)
    precision = metrics.precision_score(y_train, y_pred_train)
    recall = metrics.recall_score(y_train, y_pred_train)
    accuracy = metrics.accuracy_score(y_train, y_pred_train)
    f1_score = metrics.f1_score(y_train, y_pred_train)
    print("train roc_auc_score = %0.4f " % (roc_auc))
    print("train Precision =", precision)
    print("train Recall =", recall)
    print("train Accuracy =", accuracy)
    print("train F1score =", f1_score)

    # save the scores in arrays (train)
    TRAINmeanAUC = np.append(TRAINmeanAUC, roc_auc)
    TRAINmeanPrecision = np.append(TRAINmeanPrecision, precision)
    TRAINmeanRecall = np.append(TRAINmeanRecall, recall)
    TRAINmeanAccuracy = np.append(TRAINmeanAccuracy, accuracy)
    TRAINmeanF1score = np.append(TRAINmeanF1score, f1_score)
    
# get the top 10 features selected by different models
vi_ind = list(map(feature_vote.index, heapq.nlargest(10, feature_vote)))
print(vi_ind)

# save the scores to csv files
dataFrame = pd.DataFrame({'AUC': meanAUC, 'Precision': meanPrecision, 'Recall': meanRecall, 'Accuracy': meanAccuracy, 'F1_score': meanF1score,
    'trainAUC': TRAINmeanAUC, 'trainPrecision': TRAINmeanPrecision, 'trainRecall': TRAINmeanRecall, 'trainAccuracy': TRAINmeanAccuracy,
    'trainF1score': TRAINmeanF1score})
dataFrame.to_csv('score/smote/catboost.csv', index=True, sep=',')

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
