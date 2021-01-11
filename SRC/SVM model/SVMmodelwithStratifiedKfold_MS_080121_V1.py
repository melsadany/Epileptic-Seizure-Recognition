"""
Final SVM model with stratified kfold
@author: Muhammad Elsadany
"""


import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from statistics import mean, stdev
from sklearn.preprocessing import StandardScaler

#########################################
# data retreival line

Data = pd.read_csv("data2.csv")
# print(Data.shape)
# print(Data.head(5))


# splitting the table into X and y 
X_tot = Data.values[:,1:179]
# print(X_tot)
Y_tot = Data.values[:,179]
# print(Y_tot)

########################################
# checking null data cells if any 

# print(Data.isnull().sum())

########################################

X_tot = pd.DataFrame(X_tot)


# define standard scaler
scaler = StandardScaler()



########################################
#splitting training data into training and validation
# and using the svm model with each stratified fold


model = svm.SVC()

skf = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)


lst_accu_stratified = [] 

for train_index, test_index in skf.split(X_tot, Y_tot): 
    x_train_fold, x_test_fold = X_tot.iloc[train_index], X_tot.iloc[test_index] 
    y_train_fold, y_test_fold = Y_tot[train_index], Y_tot[test_index] 
    x_train_fold = scaler.fit_transform(x_train_fold)
    model.fit(x_train_fold, y_train_fold) 
    x_test_fold = scaler.transform(x_test_fold)
    lst_accu_stratified.append(model.score(x_test_fold, y_test_fold)) 


# Print the output. 

print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%') 
print('\nStandard Deviation is:', stdev(lst_accu_stratified)) 



