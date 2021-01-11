import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


#########################################
# data retreival line

Data = pd.read_csv("data.csv")
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
# Preprocessing step
# normalization 

X_tot = pd.DataFrame(X_tot)

########################################
#splitting data into training and testing

X_train_tot, X_test, Y_train_tot, Y_test =  train_test_split(X_tot, 
        Y_tot, test_size=0.3, shuffle=True, random_state=123, stratify=Y_tot)

# print(X_train_tot)
# print(X_train_tot.shape)
# print(X_test)
# print(X_test.shape)
# print(Y_train_tot)
# print(Y_train_tot.shape)
# print(Y_test)
# print(Y_test.shape)

########################################
#splitting training data into training and validation

X_train, X_valid, Y_train, Y_valid =  train_test_split(X_train_tot, Y_train_tot, 
        test_size=0.3, shuffle=True, random_state=123, stratify=Y_train_tot)

# print(X_train)
# print(X_train.shape)
# print(Y_train)
# print(Y_train.shape)
# print(X_valid)
# print(X_valid.shape)
# print(Y_valid)
# print(Y_valid.shape)

########################################
# SVM model


# define standard scaler
scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)


# model = svm.SVC()
# model = svm.SVC(kernel='linear')
model = svm.SVC(kernel='poly')

svm1 = model.fit(X_train, Y_train)

Y_train_pred = svm1.predict(X_train)

print(confusion_matrix(Y_train,Y_train_pred))
print(classification_report(Y_train,Y_train_pred))

##################################
# For validation data
X_valid =  scaler.transform(X_valid)

Y_valid_pred = svm1.predict(X_valid)

print(confusion_matrix(Y_valid,Y_valid_pred))
print(classification_report(Y_valid,Y_valid_pred))

##################################
# For testing data

X_test =  scaler.transform(X_test)
Y_test_pred = svm1.predict(X_test)

print(confusion_matrix(Y_test,Y_test_pred))
print(classification_report(Y_test,Y_test_pred))

#################################

print("#################################")

#################################

scaler2 = StandardScaler()

scaler2.fit(X_train_tot)
X_train_tot = scaler2.transform(X_train_tot)

svm2 = model.fit(X_train_tot, Y_train_tot)

Y_train_tot_pred = svm1.predict(X_train_tot)

print(confusion_matrix(Y_train_tot,Y_train_tot_pred))
print(classification_report(Y_train_tot,Y_train_tot_pred))

##################################
# For testing data

X_test = scaler2.transform(X_test)
Y_test_pred2 = svm2.predict(X_test)

print(confusion_matrix(Y_test,Y_test_pred2))
print(classification_report(Y_test,Y_test_pred2))

##################################



####################################











