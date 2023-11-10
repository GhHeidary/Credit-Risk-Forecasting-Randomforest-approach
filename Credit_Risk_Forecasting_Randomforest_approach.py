# -*- coding: utf-8 -*-

#---Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
#%%---importing the dataset

df=pd.read_csv(r"D:\Weiter_Bildung\Data Science Python\Woche_4\credit_risk.csv")
df.head()

#%%---EDA(first apply info and describe)

df.info()

#%%---We do not need IDs, we are going to remove Id coulmn

df.drop("Id",axis=1,inplace=True)

#%%---Data Preprocessing and null values imputation

mean_Emp=df["Emp_length"].mean()
mean_Rate=df["Rate"].mean()
df["Emp_length"].fillna(value=mean_Emp,inplace=True)
df["Rate"].fillna(value=mean_Rate,inplace=True)
df.info()
df

#%%---split label as Y

y=df.loc[:,'Default'].values
y

#%% ---label encodinig

encoder=LabelEncoder()
y_e=encoder.fit_transform(y)

#%%---remove  Y from data sets

df.pop('Default')
df

#%%---one hot incoding

df=pd.get_dummies(df)
df

#%%---define featuer values

X=df.values

#%%---Feature Scaling

scale= StandardScaler()
x_sc=scale.fit_transform(X)

#%%---Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(x_sc, y_e, test_size = 0.20)

#%%---Training the Random Forest Classification model on the Training set

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#%%---Evaluating on Test set 

y_pred=classifier.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy)

#%%---hyperparameter optimization 

from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
space = {
    "n_estimators": hp.choice("n_estimators", [50,100, 200, 300, 400]),
    "max_depth": hp.choice('max_depth', np.arange(1, 16, dtype=int)),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
}

#---define objective function
def hyperparameter_tuning(params):
    clf = RandomForestClassifier(**params,n_jobs=-1,random_state = 0)
    acc = cross_val_score(clf, X_train, y_train, scoring="accuracy", n_jobs=-1).mean()
    return {"loss": -acc, "status": STATUS_OK}

#---Initialize trials object
trials = Trials()

best = fmin(
    fn=hyperparameter_tuning,
    space = space, 
    algo=tpe.suggest, 
    max_evals=250, 
    trials=trials
)

print("Best: {}".format(best))


#%%---Training the Random Forest Classification model on the Training set after got Best hyperparameter  Best: {'criterion': 0, 'max_depth': 11, 'n_estimators': 3}

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'gini',max_depth=10, random_state = 0)
classifier.fit(X_train, y_train)

#%%-------------------evaluation

y_pred=classifier.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy)


#%%-----------------feature importances

feature_importance=pd.DataFrame({'rfc':classifier.feature_importances_},index=df.columns)
feature_importance.sort_values(by='rfc',ascending=True,inplace=True)

index = np.arange(len(feature_importance))
fig, ax = plt.subplots(figsize=(18,8))
rfc_feature=ax.barh(index,feature_importance['rfc'],0.4,color='purple',label='Random Forest')
ax.set(yticks=index+0.4,yticklabels=feature_importance.index)

ax.legend()
plt.savefig('feature_importance_plot.png')

plt.show()


































