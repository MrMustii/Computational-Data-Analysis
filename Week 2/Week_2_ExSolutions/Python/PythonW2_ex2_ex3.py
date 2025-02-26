# -*- coding: utf-8 -*-
"""
Computer intensive datahandling exercise 2 week 2

@author: dnor
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

path = "C:\\Users\\dnor\\Desktop\\02582\\Lecture2\\S2"

mat = scipy.io.loadmat('Silhouettes.mat')
Fem = mat['Fem'].ravel() - 1 # Get rid of outer dim, -1 due to stupid matlab indexing
Male = mat['Male'].ravel() - 1
Xa = mat['Xa']
#%% Exercise 2
fig, axis = plt.subplots(2, 2)
axis[0,0].plot(Xa[Fem,:65].T, Xa[Fem, 65:].T)
axis[0,0].set_title("Female Silhouettes")

axis[0,1].plot(Xa[Male, :65].T, Xa[Male, 65:].T)
axis[0,1].set_title("Male Silhouttes")

for i in range(2):
    axis[0,i].axis('equal')
    axis[0,i].axis([-0.25, 0.25, -0.25, 0.25])

N = np.shape(Xa)[0]
y = np.zeros(N)
y[Fem] = 1
n_classes = 2

MCrep = 10
K = 5

Error = np.zeros((K, 10))

I = np.asarray([0] * N)
for j in range(MCrep):
    for n in range(N):
        I[n] = (n + 1) % K + 1
    I = I[np.random.permutation(N)]
    for i in range(1, K+1):
        X_train = Xa[i != I, :]
        y_train = y[i != I]
        X_test = Xa[i == I, :]
        y_test = y[i == I]
        '''
        Can also make test training split as;
        X_train, X_test, y_train, y_test = train_test_split(...
                Xa, y, test_size=0.33, random_state=42)
        '''
        for k in range(1,11):
            # Use Scikit KNN classifier, as you have already tried implementing it youself
            neigh = KNeighborsClassifier(n_neighbors=k, weights = 'uniform', metric = 'euclidean')
            neigh.fit(X_train, y_train)
            yhat = neigh.predict(X_test)
            
            Error[i-1, k-1] = sum(np.abs(y_test - yhat)) / len(yhat)

E = np.mean(Error, axis = 0)
axis[1,0].scatter(list(range(1,11)), E, marker = '*')
axis[1,0].axis([0, 11, 0.2, 0.6])
axis[1,0].set_title("CV test error")
axis[1,0].set_xlabel("K")
axis[1,0].set_ylabel("Error")

# ROC curve
# Compute ROC curve and ROC area for each class, here based on last cross-validation fold

fpr = dict()
tpr = dict()
roc_auc = dict()

y_score = neigh.predict_proba(X_test) # find probabilities for a specific fold case

# Make y 1-hot encoded
y_1hot = np.zeros((8, n_classes))
y_1hot[np.arange(8), y_test.astype(int)] = 1    
for i in range(n_classes): # A bit redundant here, as there only is 2 classes
    fpr[i], tpr[i], threshold = roc_curve(y_1hot[:, i], y_score[:, i])
    print (threshold)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area, class 1 (females)
lw = 2
axis[1,1].plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
axis[1,1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
axis[1,1].set_xlim([0.0, 1.0])
axis[1,1].set_ylim([0.0, 1.05])
axis[1,1].set_xlabel('False Positive Rate')
axis[1,1].set_ylabel('True Positive Rate')
axis[1,1].set_title('ROC')
axis[1,1].legend(loc="upper left")

#%% Exercise 3
# use the same data as above from the last cv loop
 
def roc_data(y_hat, y_true, cut):
    
    sensitivity = np.zeros(len(cut))
    specificity = np.zeros(len(cut))
    for index, threshold in enumerate(cut):
        num_positive = np.sum(y_true == 1)
        num_negative = np.sum(y_true == 0)
        true_positive = np.sum(y_hat[y_true==1] >= threshold)
        true_negative = np.sum(y_hat[y_true==0]  < threshold)
        
        sensitivity[index] = true_positive / num_positive
        specificity[index] = true_negative / num_negative
    
    return sensitivity, specificity

fpr = dict()
tpr = dict()
roc_auc = dict()
cutoff = np.linspace(1.0, 0.0, 10)
for i in range(n_classes): # A bit redundant here, as there only is 2 classes
    tpr[i], specificity = roc_data(y_score[:, i], y_1hot[:, i], cutoff)
    # FPR is 1 - TNR(specificity)
    fpr[i] = 1 - specificity
    roc_auc[i] = auc(fpr[i], tpr[i])



plt.figure()
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="upper left")