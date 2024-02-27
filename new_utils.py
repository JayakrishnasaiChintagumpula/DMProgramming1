"""
   Use to create your own functions for reuse 
   across the assignment

   Inside part_1_template_solution.py, 
  
     import new_utils
  
    or
  
     import new_utils as nu
"""
import numpy as np
from numpy.typing import NDArray
from typing import Type, Dict
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    cross_validate,
    KFold,
)

def scale_data(X):
   X=(X - X.min())  / (X.max() - X.min())
   return X

def remove9s(X:NDArray[np.floating], y:NDArray[np.int32]):
   nine_indx = (y == 9)

   X_90 = X[nine_indx, :]
   y_90 = y[nine_indx]

   X_90=X_90[:int((X_90.shape[0])*0.1),:]
   y_90=y_90[:int((y_90.shape[0])*0.1)]
    
   none_9= (y!=9)
   X_none = X[none_9, :]
   y_none = y[none_9]
    
   finX=np.concatenate((X_none,X_90),axis=0)
   finy=np.concatenate((y_none,y_90),axis=0)
    
   return finX, finy
   
def conf_mat_accuracy(matrix):
   """
   We need to calculate accuracy from confusion matrix.
   """
   True_positive = matrix[1, 1]  
   True_negative = matrix[0, 0]  
   Total_samples = matrix.sum()
   accuracy = (True_positive+ True_negative) / Total_samples
   return accuracy

def convert_7_0(X: NDArray[np.floating], y: NDArray[np.int32]):
   id_7=(y==7)
   id_0=(y==0)
   y[id_7]=0

   return X,y

def train_simple_classifier_with_cv(
   *,
   Xtrain: NDArray[np.floating],
   ytrain: NDArray[np.int32],
   clf: BaseEstimator,
   cv: KFold = KFold,
):
   scores = cross_validation(clf,Xtrain,ytrain, cv=cv,scoring="accuracy")
   return scores


def convert_9_1(X: NDArray[np.floating], y: NDArray[np.int32]):
   id_9=(y==9)
   id_1=(y==1)
   y[id_9]=1

   return X,y   

      
