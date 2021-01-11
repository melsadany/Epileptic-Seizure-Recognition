import pandas as pd
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
from sklearn.model_selection import GridSearchCV

#########################################
# data retreival line

Data = pd.read_csv("data.csv")
# Data = pd.read_csv("data2.csv")
# print(Data.shape)
# print(Data.head(5))


# splitting the table into X and y 
X_tot = Data.values[:,1:179]
# print(X_tot)
Y_tot = Data.values[:,179]
Y_tot=Y_tot.astype('int')
# print(Y_tot)

########################################
# checking null data cells if any 

print(Data.isnull().sum())                   

########################################
# Preprocessing step
# normalization 

X_tot = pd.DataFrame(X_tot)
pt = StandardScaler()


########################################
# Stratified k-fold
r_f = RandomForestClassifier(n_estimators = 95, random_state = 123)
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
kf.get_n_splits(X_tot, Y_tot)
i=0
for train,test in kf.split(X_tot,Y_tot):
    X_train, X_test = X_tot.iloc[train], X_tot.iloc[test]
    y_train, y_test = Y_tot[train], Y_tot[test]
    pt.fit(X_train)
    X_train_normalized=pt.transform(X_train)
    X_test_normalized=pt.transform(X_test)
    # r_f.fit(X_train_normalized, y_train)
    # y_pred = r_f.predict(X_test_normalized)    
    r_f.fit(X_train, y_train)
    y_pred = r_f.predict(X_test)
    classificationreport=classification_report(y_test,y_pred)
    i=i+1
    print("fold",i,classificationreport)


########################################
#splitting data into training and testing

X_train_tot, X_test, Y_train_tot, Y_test =  train_test_split(X_tot, 
        Y_tot, test_size=0.2, shuffle=True, random_state=123, stratify=Y_tot)

# print(X_train_tot)
# print(X_train_tot.shape)
# print(X_test)
# print(X_test.shape)
# print(Y_train_tot)
# print(Y_train_tot.shape)
# print(Y_test)
# print(Y_test.shape)

pt.fit(X_train_tot)
X_train_tot_normalized=pt.transform(X_train_tot)
X_test_normalized=pt.transform(X_test)

########################################
#splitting training data into training and validation

X_train, X_valid, Y_train, Y_valid =  train_test_split(X_train_tot, Y_train_tot, 
        test_size=0.2, shuffle=True, random_state=123, stratify=Y_train_tot)

# print(X_train)
# print(X_train.shape)
# print(Y_train)
# print(Y_train.shape)
# print(X_valid)
# print(X_valid.shape)
# print(Y_valid)
# print(Y_valid.shape)


########################################
# Random forest Classifier
rf = RandomForestClassifier(n_estimators = 95, random_state = 123)
pt.fit(X_train)
X_train_normalized=pt.transform(X_train)
X_valid_normalized=pt.transform(X_valid)


# Train the model on training data without validation
# rf.fit(X_train_tot_normalized,Y_train_tot)
rf.fit(X_train_tot,Y_train_tot)


# predicting the training and testing without validation
# y_pred=rf.predict(X_train_tot_normalized)
# y_testpred=rf.predict(X_test_normalized)
y_pred=rf.predict(X_train_tot)
y_testpred=rf.predict(X_test)

print ('Confusion_report of training', classification_report(Y_train_tot,y_pred))
print ('Confusion_report of testing', classification_report(Y_test,y_testpred))

# Train the model on training data with validation
# rf.fit(X_train_normalized,Y_train)
rf.fit(X_train,Y_train)

# predicting the validation and testing
# y_train_pred=rf.predict(X_train_normalized)
# y_valid_pred=rf.predict(X_valid_normalized)
# y_test_pred=rf.predict(X_test_normalized)

y_train_pred=rf.predict(X_train)
y_valid_pred=rf.predict(X_valid)
y_test_pred=rf.predict(X_test)

########################################
# Accuracy 
# print ('Confusion_matrix for training',confusion_matrix(Y_train,y_train_pred))
print ('Confusion_report of training', classification_report(Y_train,y_train_pred))

# print ('Confusion_matrix for Validating',confusion_matrix(Y_valid,y_valid_pred))
print ('Confusion_report of Validating', classification_report(Y_valid,y_valid_pred))

# print ('Confusion_matrix for testing',confusion_matrix(Y_test,y_test_pred))
print ('Confusion_report of tesing', classification_report(Y_test,y_test_pred))


########################################

# the Grid Search for the best no of trees
# n_estimators = range(10, 100, 5)
# param_grid = dict(n_estimators=n_estimators)
# # Create the parameter grid based on the results of random search 

# # kfold = StratifiedKFold(n_splits scoring="neg_log_loss", n_jobs=-1, cv=kfold)
# # result = grid_search.fit(X, label_encoded_y)
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,cv=5, n_jobs = -1)
# grid_search.fit(X_train, Y_train)
# best_grid = grid_search.best_estimator_
# print(best_grid)
########################################
#feature importance (interpretability)
sorted_idx = rf.feature_importances_.argsort()
feature_list=X_tot.columns
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature,importance) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances[:20]];
