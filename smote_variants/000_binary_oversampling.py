import numpy as np
import matplotlib.pyplot as plt
import smote_variants as sv
import imbalanced_databases as imbd
import pandas as pd

# loading the dataset
#dataset= imbd.load_iris0()
#X, y= dataset['data'], dataset['target']
df = pd.read_csv("dev.csv")
data =np.array(df)
X=data[:,:-1]
y=data[:,-1]

imporInd1=5 -1
imporInd2=17 -1

plt.figure(figsize=(10, 5))
plt.scatter(X[y == 0][:,imporInd1], X[y == 0][:,imporInd2], label='majority class', c='orange')
plt.scatter(X[y == 1][:,imporInd1], X[y == 1][:,imporInd2], label='minority class', c='olive')
plt.title('original dataset')
plt.xlabel('coordinate 0')
plt.ylabel('coordinate 1')
plt.legend()
plt.savefig('pic1.jpg')

# printing the number of samples

print('majority class: %d' % np.sum(y == 0))
print('minority class: %d' % np.sum(y == 1))

# ## Oversampling
# 
# The oversampling is carried out by instantiating any oversampler implemented in the package and calling the ```sample``` function.

oversampler= sv.distance_SMOTE()
X_samp, y_samp= oversampler.sample(X, y)
X_samp=np.round(X_samp)
y_samp=np.round(y_samp)

# ## Illustrating the oversampled dataset
# 
# The oversampled dataset is illustrated by printing the number of samples after oversampling. The newly generated samples can be filtered by removing the first ```len(X)``` elements of the oversampled dataset. Note that this filtering works only with *extensive* techniques, namely, with oversamplers which only add samples to the dataset. There are multiple oversamplers which also remove noisy samples, with these oversamplers there is no way to filter newly generated samples.

# printing the number of samples
print('majority class: %d' % np.sum(y_samp == 0))
print('minority class: %d' % np.sum(y_samp == 1))

# filtering new samples
X_samp, y_samp= X_samp[len(X):], y_samp[len(y):]


# printing the number of new samples
print('majority new samples: %d' % np.sum(y_samp == 0))
print('minority new samples: %d' % np.sum(y_samp == 1))

plt.figure(figsize=(10, 5))

plt.scatter(X[y == 0][:,imporInd1], X[y == 0][:,imporInd2], label='majority class', c='orange', marker='o')
plt.scatter(X[y == 1][:,imporInd1], X[y == 1][:,imporInd2], label='minority class', c='olive', marker='o')
plt.scatter(X_samp[y_samp == 1][:,imporInd1], X_samp[y_samp == 1][:,imporInd2], label='new minority samples', c='olive', marker='x')
plt.title('oversampled dataset')
plt.xlabel('coordinate 0')
plt.ylabel('coordinate 1')
plt.savefig('pic2.jpg')
plt.show()


